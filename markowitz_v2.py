"""
Otimizador de Carteira v2.0 — Markowitz para Leigos
====================================================
- Regras de risco PRÉ-ESTABELECIDAS:
    * Máximo 20% por ativo
    * Máximo 35% por setor
- Modo Admin: cadastra a carteira recomendada da casa (pool de ações por setor)
- Modo Cliente: insere a carteira atual e recebe diagnóstico + 3 sugestões
- Substituição automática: se um setor estourou, troca por ações de setores
  ausentes, vindas da carteira recomendada do admin
- 3 perfis de sugestão: Mais Eficiência (Sharpe), Mais Segurança (menor risco),
  Mais Retorno
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# CONSTANTES — REGRAS DA CASA (não editáveis pelo cliente)
# ============================================================
TRADING_DAYS = 252
LIMITE_POR_ATIVO = 0.20          # 20% máximo por ativo
LIMITE_POR_SETOR = 0.35          # 35% máximo por setor
ANOS_HISTORICO = 10              # histórico fixo de 10 anos
TAXA_RF_PADRAO = 0.105           # fallback se admin ainda não cadastrou
ARQUIVO_SETOR_PADRAO = "ClassifSetorial.xlsx"
ARQUIVO_CARTEIRA_ADMIN = "carteira_admin.json"
ARQUIVO_CONFIG_ADMIN = "config_admin.json"

# Paleta cores
COR_OK = "#16a34a"        # verde
COR_ALERTA = "#f59e0b"    # âmbar
COR_RUIM = "#dc2626"      # vermelho
COR_NEUTRA = "#3b82f6"    # azul

# ============================================================
# NÚCLEO QUANTITATIVO (Markowitz)
# ============================================================
class CarteiraMarkowitz:
    def __init__(self, tickers, start="2020-01-01", end=None):
        self.tickers_originais = [t.strip().upper().replace(".SA", "") for t in tickers if t.strip()]
        self.start = start
        self.end = end

        self.precos = None
        self.retornos = None
        self.retorno_medio_anual = None
        self.cov_anual = None
        self.correlacao = None
        self.estatisticas_ativos = None
        self.nomes_ativos = None
        self.falhas_download = []
        self.taxa_livre_risco_anual = 0.0

    def baixar_um_ticker(self, ticker, tentativas=3, pausa=2):
        ticker_yf = f"{ticker}.SA"
        ultimo_erro = None

        for tentativa in range(1, tentativas + 1):
            try:
                df = yf.download(
                    ticker_yf,
                    start=self.start,
                    end=self.end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=20
                )

                if df is None or df.empty:
                    raise ValueError(f"Sem dados retornados para {ticker_yf}")

                if isinstance(df.columns, pd.MultiIndex):
                    if "Close" in df.columns.get_level_values(0):
                        serie = df["Close"].copy()
                    else:
                        serie = df.xs("Close", axis=1, level=0).copy()
                else:
                    if "Close" in df.columns:
                        serie = df["Close"].copy()
                    else:
                        serie = df.iloc[:, 0].copy()

                if isinstance(serie, pd.DataFrame):
                    serie = serie.iloc[:, 0]

                serie = pd.to_numeric(serie, errors="coerce").dropna()

                if serie.empty:
                    raise ValueError(f"Série vazia para {ticker_yf}")

                serie.name = ticker
                serie.index = pd.to_datetime(serie.index)
                return serie.sort_index()

            except Exception as e:
                ultimo_erro = e
                if tentativa < tentativas:
                    time.sleep(pausa)

        raise RuntimeError(f"Falha ao baixar {ticker_yf}: {ultimo_erro}")

    def baixar_dados(self):
        """
        Baixa dados de cada ticker. Tickers com histórico curto (IPO recente, p.ex.)
        NÃO derrubam os outros — eles são apenas registrados em falhas_download e
        descartados. O DataFrame final começa na primeira data em que TODOS os
        tickers sobreviventes têm preço.
        """
        series = []
        falhas = []

        for ticker in self.tickers_originais:
            try:
                s = self.baixar_um_ticker(ticker)
                series.append(s)
            except Exception as e:
                falhas.append((ticker, str(e)))

        if not series:
            self.falhas_download = falhas
            raise ValueError("Nenhum ticker retornou dados. Verifique tickers / conexão.")

        precos = pd.concat(series, axis=1).sort_index()
        precos.index = pd.to_datetime(precos.index)
        precos = precos.dropna(how="all")

        # Estratégia: para CADA ticker, calcular há quantos dias úteis ele tem dados
        # (a partir da primeira cotação dele). Tickers com menos de N_MIN_DIAS úteis
        # são descartados (e reportados em falhas_download).
        N_MIN_DIAS = 252  # ~1 ano útil
        tickers_validos = []
        for col in precos.columns:
            serie = precos[col].dropna()
            if len(serie) >= N_MIN_DIAS:
                tickers_validos.append(col)
            else:
                falhas.append((col, f"Histórico curto: só {len(serie)} pregões disponíveis (mín. {N_MIN_DIAS})"))

        self.falhas_download = falhas

        if len(tickers_validos) < 2:
            raise ValueError(
                f"Apenas {len(tickers_validos)} ativo(s) com histórico suficiente. "
                f"Precisa de no mínimo 2. Tickers descartados por histórico curto: " +
                ", ".join([t for t, _ in falhas if "Histórico curto" in _])
            )

        # Agora alinha: pega o intervalo onde TODOS os tickers válidos têm dado
        precos = precos[tickers_validos].dropna()

        if len(precos) < N_MIN_DIAS:
            # Algum ticker do conjunto tem início muito tarde -> remove o "mais novo"
            # iterativamente até sobrar histórico suficiente
            while len(precos) < N_MIN_DIAS and len(tickers_validos) > 2:
                # Identifica o ticker que começa mais tarde
                primeiras_datas = {t: precos[t].first_valid_index() if precos[t].notna().any()
                                   else None for t in tickers_validos}
                # Recalcula sem esse ticker
                base_precos = pd.concat(series, axis=1).sort_index()
                base_precos = base_precos[[c for c in base_precos.columns if c in tickers_validos]]
                base_precos = base_precos.dropna(how="all")
                primeiras_datas = {t: base_precos[t].first_valid_index() for t in tickers_validos}
                ticker_mais_novo = max(primeiras_datas, key=primeiras_datas.get)
                tickers_validos.remove(ticker_mais_novo)
                falhas.append((ticker_mais_novo,
                               f"Removido para preservar histórico (começou em {primeiras_datas[ticker_mais_novo].date()})"))
                precos = base_precos[tickers_validos].dropna()
            self.falhas_download = falhas

        if precos.shape[1] < 2 or len(precos) < N_MIN_DIAS:
            raise ValueError(
                f"Após alinhamento sobraram só {precos.shape[1]} ativo(s) "
                f"com {len(precos)} pregões. Tente reduzir o histórico ou usar tickers mais antigos."
            )

        self.precos = precos
        self.nomes_ativos = list(precos.columns)
        return self.precos

    def calcular_metricas(self, taxa_livre_risco=0.0):
        retornos = self.precos.pct_change().dropna()

        if retornos.empty:
            raise ValueError("Não foi possível calcular retornos.")

        self.retornos = retornos
        self.retorno_medio_anual = retornos.mean() * TRADING_DAYS
        self.cov_anual = retornos.cov() * TRADING_DAYS
        self.correlacao = retornos.corr()
        self.taxa_livre_risco_anual = float(taxa_livre_risco)

        vol = retornos.std() * np.sqrt(TRADING_DAYS)
        excesso = self.retorno_medio_anual - self.taxa_livre_risco_anual
        sharpe = np.where(vol > 0, excesso / vol, np.nan)

        self.estatisticas_ativos = pd.DataFrame({
            "retorno_anual": self.retorno_medio_anual,
            "volatilidade_anual": vol,
            "sharpe": sharpe
        }).sort_values("sharpe", ascending=False)

    def estatisticas_carteira(self, pesos, taxa_livre_risco=0.0):
        pesos = np.array(pesos, dtype=float)
        if not np.isclose(pesos.sum(), 1.0, atol=1e-4):
            pesos = pesos / pesos.sum() if pesos.sum() > 0 else pesos
        retorno = float(np.dot(pesos, self.retorno_medio_anual.values))
        vol = float(np.sqrt(np.dot(pesos.T, np.dot(self.cov_anual.values, pesos))))
        sharpe = (retorno - taxa_livre_risco) / vol if vol > 0 else np.nan
        return {
            "pesos": pesos,
            "retorno_anual": retorno,
            "volatilidade_anual": vol,
            "risco_anual": vol,
            "sharpe": sharpe
        }

    def _restricoes(self, retorno_alvo=None, mapa_setor=None, limite_setor=None):
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if retorno_alvo is not None:
            cons.append({"type": "eq",
                         "fun": lambda w, r=retorno_alvo: np.dot(w, self.retorno_medio_anual.values) - r})
        if mapa_setor and limite_setor is not None:
            setores = pd.Series(mapa_setor).reindex(self.nomes_ativos).fillna("Sem setor")
            for setor in setores.unique():
                idx = np.where(setores.values == setor)[0]
                cons.append({"type": "ineq",
                             "fun": lambda w, idx=idx, lim=limite_setor: lim - np.sum(w[idx])})
        return cons

    def _verificar_factibilidade(self, peso_maximo, mapa_setor, limite_setor):
        """
        Verifica se o conjunto de restrições admite alguma solução.
        - n ativos com peso máx p_max -> capacidade = n * p_max (precisa ser >= 1)
        - se há mapa setor com limite l_setor: soma_setores(min(qtd_ativos*p_max, l_setor))
          precisa ser >= 1 também.
        Retorna (factivel: bool, motivo: str).
        """
        n = len(self.nomes_ativos)
        if n * peso_maximo < 1.0 - 1e-9:
            return False, (f"Com {n} ativos e peso máximo {peso_maximo:.0%} por ativo, "
                           f"a capacidade total é {n*peso_maximo:.0%}, menor que 100%. "
                           f"Adicione mais ativos ao universo.")
        if mapa_setor and limite_setor is not None:
            setores = pd.Series(mapa_setor).reindex(self.nomes_ativos).fillna("Sem setor")
            capacidade = 0.0
            detalhes = []
            for setor in setores.unique():
                qtd = int((setores == setor).sum())
                cap_setor = min(qtd * peso_maximo, limite_setor)
                capacidade += cap_setor
                detalhes.append(f"{setor}: {qtd} ativos -> cap {cap_setor:.0%}")
            if capacidade < 1.0 - 1e-9:
                return False, (f"As restrições deixam apenas {capacidade:.0%} de capacidade "
                               f"total (precisa de 100%). Detalhe: " + "; ".join(detalhes) +
                               ". Solução: incluir ações de mais setores.")
        return True, ""

    def otimizar(self, modo, taxa_livre_risco=0.0, peso_maximo=LIMITE_POR_ATIVO,
                 mapa_setor=None, limite_setor=LIMITE_POR_SETOR):
        """modo: 'min_var' | 'max_sharpe' | 'max_retorno'"""
        factivel, motivo = self._verificar_factibilidade(peso_maximo, mapa_setor, limite_setor)
        if not factivel:
            raise ValueError(f"Restrições infactíveis: {motivo}")

        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = tuple((0, peso_maximo) for _ in range(n))
        cons = self._restricoes(mapa_setor=mapa_setor, limite_setor=limite_setor)

        if modo == "min_var":
            fun = lambda w: np.dot(w.T, np.dot(self.cov_anual.values, w))
        elif modo == "max_sharpe":
            def fun(w):
                ret = np.dot(w, self.retorno_medio_anual.values)
                v = np.sqrt(np.dot(w.T, np.dot(self.cov_anual.values, w)))
                return 1e10 if v <= 0 else -((ret - taxa_livre_risco) / v)
        elif modo == "max_retorno":
            fun = lambda w: -np.dot(w, self.retorno_medio_anual.values)
        else:
            raise ValueError(f"Modo desconhecido: {modo}")

        res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-9})
        if not res.success:
            # tenta ponto inicial alternativo
            x0_alt = np.random.dirichlet(np.ones(n))
            res = minimize(fun, x0_alt, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 1000, "ftol": 1e-9})
            if not res.success:
                raise RuntimeError(f"Otimização falhou ({modo}): {res.message}")

        return self.estatisticas_carteira(res.x, taxa_livre_risco)

    def fronteira_eficiente(self, n_pontos=40, peso_maximo=LIMITE_POR_ATIVO,
                            mapa_setor=None, limite_setor=LIMITE_POR_SETOR):
        retornos_alvo = np.linspace(self.retorno_medio_anual.min(),
                                     self.retorno_medio_anual.max(), n_pontos)
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = tuple((0, peso_maximo) for _ in range(n))
        riscos, rets = [], []

        for r_alvo in retornos_alvo:
            cons = self._restricoes(retorno_alvo=r_alvo, mapa_setor=mapa_setor,
                                     limite_setor=limite_setor)
            res = minimize(lambda w: np.dot(w.T, np.dot(self.cov_anual.values, w)),
                           x0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 300})
            if res.success:
                v = np.sqrt(np.dot(res.x.T, np.dot(self.cov_anual.values, res.x)))
                riscos.append(v)
                rets.append(r_alvo)

        return pd.DataFrame({"risco": riscos, "retorno": rets})


# ============================================================
# UTILITÁRIOS DE CLASSIFICAÇÃO SETORIAL
# ============================================================
def normalizar_ticker(ticker):
    """
    Normaliza o ticker para uso geral (download, exibição).
    Preserva números: 'petr4.sa' -> 'PETR4', '  itub4 ' -> 'ITUB4'.
    """
    return str(ticker).upper().strip().replace(".SA", "")


def normalizar_codigo(codigo):
    """
    Reduz a apenas letras, para CASAR com a tabela de classificação setorial.
    'PETR4' -> 'PETR', 'KLBN11' -> 'KLBN'. NÃO usar para download — perde
    informação de classe da ação.
    """
    s = normalizar_ticker(codigo)
    letras = "".join(ch for ch in s if ch.isalpha())
    return letras if letras else s


@st.cache_data(show_spinner=False)
def carregar_classificacao(arquivo):
    df = pd.read_excel(arquivo)
    df = df.rename(columns={df.columns[0]: "SETOR",
                             df.columns[1]: "SUBSETOR",
                             df.columns[2]: "CODIGO"})
    df["CODIGO"] = df["CODIGO"].apply(normalizar_codigo)
    df["SETOR"] = df["SETOR"].fillna("Sem setor")
    df["SUBSETOR"] = df["SUBSETOR"].fillna("Sem subsetor")
    df = df[df["CODIGO"].notna()]
    df = df[df["CODIGO"] != "CÓDIGO"]
    df = df.drop_duplicates(subset=["CODIGO"])
    return df


def setor_de(ticker, classificacao_df):
    if classificacao_df is None:
        return "Sem setor"
    cod = normalizar_codigo(ticker)
    linha = classificacao_df[classificacao_df["CODIGO"] == cod]
    if linha.empty:
        return "Sem setor"
    return linha.iloc[0]["SETOR"]


def montar_mapa_setor(ativos, classificacao_df):
    return {a: setor_de(a, classificacao_df) for a in ativos}


def exposicao_setorial(ativos, pesos, classificacao_df):
    df = pd.DataFrame({"ativo": ativos, "peso": pesos})
    df["SETOR"] = df["ativo"].apply(lambda x: setor_de(x, classificacao_df))
    expo = (df.groupby("SETOR")["peso"].sum()
              .sort_values(ascending=False)
              .reset_index()
              .rename(columns={"peso": "peso_setor"}))
    expo["acima_limite"] = expo["peso_setor"] > LIMITE_POR_SETOR + 1e-8
    return df, expo


# ============================================================
# DIAGNÓSTICO — checagem de problemas da carteira atual
# ============================================================
def diagnosticar_carteira(ativos, pesos, classificacao_df):
    """Retorna lista de problemas em linguagem leiga."""
    problemas = []
    pesos = np.array(pesos)

    # 1) Concentração por ativo
    for at, p in zip(ativos, pesos):
        if p > LIMITE_POR_ATIVO + 1e-6:
            problemas.append({
                "tipo": "concentracao_ativo",
                "severidade": "alta",
                "icone": "⚠️",
                "titulo": f"{at} pesa demais",
                "descricao": (f"A ação **{at}** representa **{p:.1%}** da carteira. "
                              f"Pela regra da casa, nenhum ativo deve passar de **{LIMITE_POR_ATIVO:.0%}**. "
                              f"Concentrar muito em uma ação só amplia o risco de perdas pesadas se ela cair."),
                "valor": p
            })

    # 2) Concentração por setor
    if classificacao_df is not None:
        _, expo = exposicao_setorial(ativos, pesos, classificacao_df)
        for _, row in expo.iterrows():
            if row["acima_limite"]:
                problemas.append({
                    "tipo": "concentracao_setor",
                    "severidade": "alta",
                    "icone": "🏭",
                    "titulo": f"Setor {row['SETOR']} concentrado",
                    "descricao": (f"O setor **{row['SETOR']}** soma **{row['peso_setor']:.1%}** da carteira. "
                                  f"O limite recomendado é **{LIMITE_POR_SETOR:.0%}**. "
                                  f"Quando um setor inteiro vai mal (juros, regulação, commodities), "
                                  f"todas essas ações tendem a cair juntas."),
                    "valor": row["peso_setor"],
                    "setor": row["SETOR"]
                })

    # 3) Diversificação geral
    n_ativos_relevantes = sum(1 for p in pesos if p > 0.01)
    if n_ativos_relevantes < 5:
        problemas.append({
            "tipo": "poucos_ativos",
            "severidade": "media",
            "icone": "🎯",
            "titulo": "Poucos ativos na carteira",
            "descricao": (f"Sua carteira tem apenas **{n_ativos_relevantes}** ativos relevantes. "
                          f"Para diluir bem o risco, costuma-se sugerir pelo menos **5-8 ativos** "
                          f"de setores diferentes."),
        })

    # 4) Soma diferente de 100%
    soma = sum(pesos)
    if not np.isclose(soma, 1.0, atol=1e-3):
        problemas.append({
            "tipo": "soma_invalida",
            "severidade": "alta",
            "icone": "🧮",
            "titulo": "Pesos não somam 100%",
            "descricao": f"A soma dos pesos é **{soma:.2%}**. Ajuste para totalizar **100%**.",
        })

    return problemas


# ============================================================
# CARTEIRA RECOMENDADA DA CASA (ADMIN)
# ============================================================
def carregar_carteira_admin():
    p = Path(ARQUIVO_CARTEIRA_ADMIN)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def salvar_carteira_admin(lista):
    Path(ARQUIVO_CARTEIRA_ADMIN).write_text(
        json.dumps(lista, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def carregar_config_admin():
    """Configurações gerais cadastradas pelo admin (taxa livre de risco etc.)."""
    p = Path(ARQUIVO_CONFIG_ADMIN)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def salvar_config_admin(config):
    Path(ARQUIVO_CONFIG_ADMIN).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ============================================================
# SUBSTITUIÇÃO SETORIAL — sugere ações da carteira admin para
# preencher setores ausentes quando a carteira atual está concentrada
# ============================================================
def sugerir_substituicoes_setoriais(ativos_cliente, pesos_cliente, carteira_admin,
                                     classificacao_df):
    """
    Se algum setor do cliente está acima do limite, busca na carteira admin
    ações de setores QUE O CLIENTE NÃO TEM, para diversificar.
    Retorna lista de tickers a serem ADICIONADOS ao universo de otimização.
    """
    if classificacao_df is None or not carteira_admin:
        return [], []

    setores_cliente = set()
    for at in ativos_cliente:
        setores_cliente.add(setor_de(at, classificacao_df))

    _, expo_cli = exposicao_setorial(ativos_cliente, pesos_cliente, classificacao_df)
    setores_estourados = set(expo_cli[expo_cli["acima_limite"]]["SETOR"].tolist())

    # Filtra carteira admin: só ações cujo setor o cliente NÃO TEM.
    # IMPORTANTE: preservamos o ticker completo (PETR4, não PETR), pois ele será
    # usado para download e exibição. O lookup setorial via setor_de() já lida
    # com o truncamento internamente.
    candidatos = []
    racional = []
    codigos_cliente = {normalizar_codigo(a) for a in ativos_cliente}
    for ticker in carteira_admin:
        ticker_norm = normalizar_ticker(ticker)            # mantém número
        codigo_setorial = normalizar_codigo(ticker_norm)   # só letras p/ comparar
        setor = setor_de(ticker_norm, classificacao_df)
        if setor not in setores_cliente and setor != "Sem setor":
            if codigo_setorial not in codigos_cliente:
                candidatos.append(ticker_norm)
                racional.append({
                    "ticker": ticker_norm,
                    "setor_novo": setor,
                    "motivo": (f"Setor **{setor}** está ausente da sua carteira. "
                               f"Adicionar **{ticker_norm}** ajuda a diversificar.")
                })

    return candidatos, racional


# ============================================================
# HELPERS DE FORMATAÇÃO E VISUAL
# ============================================================
def pct(x, casas=1):
    return f"{x*100:.{casas}f}%"


def card_metric(col, label, value, help_text=None, delta=None, delta_color="normal"):
    col.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


def grafico_pizza_ativos(ativos, pesos, titulo):
    df = pd.DataFrame({"ativo": ativos, "peso": pesos})
    df = df[df["peso"] > 0.001].sort_values("peso", ascending=False)
    if df.empty:
        return go.Figure()
    fig = px.pie(df, names="ativo", values="peso", title=titulo, hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      hovertemplate="<b>%{label}</b><br>Peso: %{percent}<extra></extra>")
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(t=50, b=20, l=20, r=20))
    return fig


def grafico_pizza_setor(ativos, pesos, classificacao_df, titulo):
    _, expo = exposicao_setorial(ativos, pesos, classificacao_df)
    expo_plot = expo[expo["peso_setor"] > 0.001]
    if expo_plot.empty:
        return go.Figure()
    fig = px.pie(expo_plot, names="SETOR", values="peso_setor", title=titulo, hole=0.45)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(t=50, b=20, l=20, r=20))
    return fig


def grafico_barras_setor_com_limite(ativos_atual, pesos_atual, ativos_sug, pesos_sug,
                                     classificacao_df, titulo="Concentração por setor"):
    _, expo_a = exposicao_setorial(ativos_atual, pesos_atual, classificacao_df)
    _, expo_s = exposicao_setorial(ativos_sug, pesos_sug, classificacao_df)

    setores = sorted(set(expo_a["SETOR"]).union(set(expo_s["SETOR"])))
    a = expo_a.set_index("SETOR")["peso_setor"].reindex(setores).fillna(0)
    s = expo_s.set_index("SETOR")["peso_setor"].reindex(setores).fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=setores, y=a.values, name="Atual",
                         marker_color="#94a3b8",
                         hovertemplate="<b>%{x}</b><br>Atual: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=setores, y=s.values, name="Sugerida",
                         marker_color=COR_NEUTRA,
                         hovertemplate="<b>%{x}</b><br>Sugerida: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=LIMITE_POR_SETOR, line_dash="dash", line_color=COR_RUIM,
                  annotation_text=f"Limite {LIMITE_POR_SETOR:.0%}",
                  annotation_position="top right")
    fig.update_layout(barmode="group", title=titulo,
                      yaxis_title="Peso do setor", yaxis_tickformat=".0%",
                      template="plotly_white", height=420,
                      margin=dict(t=50, b=80, l=20, r=20))
    return fig


def grafico_risco_retorno(stats_atual, sugestoes, fronteira):
    fig = go.Figure()

    if not fronteira.empty:
        fig.add_trace(go.Scatter(
            x=fronteira["risco"], y=fronteira["retorno"],
            mode="lines", name="Fronteira eficiente",
            line=dict(color="#cbd5e1", width=2, dash="dot"),
            hovertemplate="Risco: %{x:.1%}<br>Retorno: %{y:.1%}<extra></extra>"
        ))

    cores = {"Mais Eficiência": "#2563eb",
             "Mais Segurança": COR_OK,
             "Mais Retorno": "#9333ea"}
    simbolos = {"Mais Eficiência": "star",
                "Mais Segurança": "diamond",
                "Mais Retorno": "triangle-up"}

    for nome, s in sugestoes.items():
        fig.add_trace(go.Scatter(
            x=[s["risco_anual"]], y=[s["retorno_anual"]],
            mode="markers+text", name=nome,
            marker=dict(size=20, symbol=simbolos[nome], color=cores[nome],
                        line=dict(width=2, color="white")),
            text=[nome], textposition="top center",
            hovertemplate=(f"<b>{nome}</b><br>"
                           f"Risco: %{{x:.1%}}<br>Retorno: %{{y:.1%}}<br>"
                           f"Sharpe: {s['sharpe']:.2f}<extra></extra>")
        ))

    if stats_atual is not None:
        fig.add_trace(go.Scatter(
            x=[stats_atual["risco_anual"]], y=[stats_atual["retorno_anual"]],
            mode="markers+text", name="Sua carteira hoje",
            marker=dict(size=18, symbol="x-thin", color=COR_RUIM,
                        line=dict(width=4, color=COR_RUIM)),
            text=["Você está aqui"], textposition="bottom center",
            textfont=dict(color=COR_RUIM, size=11),
            hovertemplate=("<b>Sua carteira</b><br>"
                           "Risco: %{x:.1%}<br>Retorno: %{y:.1%}<extra></extra>")
        ))

    fig.update_layout(
        title="Risco × Retorno — onde está sua carteira e para onde podemos levar",
        xaxis_title="Risco (volatilidade anual)",
        yaxis_title="Retorno esperado anual",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    return fig


def tabela_mudancas(ativos, pesos_atual, pesos_sug, classificacao_df):
    """Tabela amigável: o que aumentar, reduzir, manter, comprar, vender."""
    df = pd.DataFrame({
        "Ativo": ativos,
        "Setor": [setor_de(a, classificacao_df) for a in ativos],
        "Hoje": pesos_atual,
        "Sugerido": pesos_sug,
    })
    df["Mudança"] = df["Sugerido"] - df["Hoje"]

    def acao_label(row):
        if row["Hoje"] < 0.001 and row["Sugerido"] > 0.005:
            return "🟢 COMPRAR"
        if row["Hoje"] > 0.005 and row["Sugerido"] < 0.001:
            return "🔴 VENDER"
        if row["Mudança"] > 0.01:
            return "↑ Aumentar"
        if row["Mudança"] < -0.01:
            return "↓ Reduzir"
        return "= Manter"

    df["Ação"] = df.apply(acao_label, axis=1)
    df = df.sort_values("Sugerido", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="Otimizador de Carteira v2.0",
                   page_icon="📊", layout="wide")

# Cabeçalho
st.markdown("""
<div style='background: linear-gradient(90deg, #1e3a8a, #3b82f6);
            padding: 24px; border-radius: 12px; margin-bottom: 8px;'>
  <h1 style='color: white; margin: 0;'>📊 Otimizador de Carteira</h1>
  <p style='color: #dbeafe; margin: 4px 0 0 0; font-size: 16px;'>
    Diagnóstico simples e sugestões inteligentes para sua carteira de ações
  </p>
</div>
""", unsafe_allow_html=True)

# ----------- SIDEBAR: seletor de modo -----------
with st.sidebar:
    st.markdown("### 👤 Modo de uso")
    modo_uso = st.radio(
        "Selecione",
        ["Cliente — analisar minha carteira", "Admin — cadastrar carteira da casa"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📜 Regras da casa")
    st.markdown(f"""
    - 🎯 **Máx. {LIMITE_POR_ATIVO:.0%}** por ativo
    - 🏭 **Máx. {LIMITE_POR_SETOR:.0%}** por setor
    - 📅 Histórico: **{ANOS_HISTORICO} anos**
    """)
    st.caption("Essas regras protegem o cliente de concentrações arriscadas.")

# Carrega classificação setorial — arquivo gerenciado exclusivamente pelo Admin.
# O cliente não vê nem tem acesso ao upload desse arquivo.
caminho_setor = Path(ARQUIVO_SETOR_PADRAO)
classificacao_df = None
if caminho_setor.exists():
    try:
        classificacao_df = carregar_classificacao(str(caminho_setor))
    except Exception as e:
        st.sidebar.error(f"Erro interno ao ler classificação setorial: {e}")


# ============================================================
# MODO ADMIN
# ============================================================
if modo_uso.startswith("Admin"):
    st.subheader("🛠️ Cadastro da carteira recomendada da casa")
    st.markdown(
        "Liste aqui as ações que a casa **recomenda** para clientes. "
        "Quando um cliente tiver muita concentração em um setor, o sistema vai pegar "
        "ações desta lista — de **setores que o cliente não tem** — para sugerir como "
        "substituições e diversificar a carteira."
    )

    # Upload e salvamento do arquivo setorial — exclusivo do Admin
    if classificacao_df is None:
        st.info("📂 O arquivo de classificação setorial ainda não está configurado. "
                "Faça o upload abaixo para ativá-lo.")

    # ---------- Configurações gerais (taxa livre de risco) ----------
    config_admin = carregar_config_admin()
    rf_atual = float(config_admin.get("taxa_livre_risco", TAXA_RF_PADRAO))

    with st.expander("⚙️ Configurações da casa", expanded=False):
        st.caption(
            "Estas configurações são aplicadas a todos os clientes automaticamente. "
            "Eles não veem nem alteram esses valores."
        )
        col_rf1, col_rf2 = st.columns([2, 1])
        with col_rf1:
            nova_rf = st.number_input(
                "💰 Taxa livre de risco anual (Selic)",
                min_value=0.0, max_value=0.30,
                value=rf_atual, step=0.005, format="%.4f",
                help="Usada no cálculo do índice Sharpe das carteiras dos clientes. "
                     "Atualize sempre que a Selic mudar."
            )
        with col_rf2:
            st.metric("Em uso hoje", f"{rf_atual*100:.2f}%")

        if st.button("💾 Salvar configurações"):
            config_admin["taxa_livre_risco"] = float(nova_rf)
            salvar_config_admin(config_admin)
            st.success(f"✅ Taxa livre de risco atualizada para {nova_rf*100:.2f}%")
            st.rerun()  # recarrega para refletir o novo valor no card "Em uso hoje"

        st.caption(f"📅 Histórico de análise: **{ANOS_HISTORICO} anos** (fixo).")

    with st.expander("📁 Arquivo de classificação setorial", expanded=(classificacao_df is None)):
        arquivo_setor_up = st.file_uploader(
            "Envie o arquivo `ClassifSetorial.xlsx`",
            type=["xlsx", "xls"],
            help="Este arquivo define a qual setor cada ação pertence. "
                 "Depois de enviado, fica salvo no servidor e o cliente nunca precisa vê-lo.",
            key="upload_setor"
        )
        if arquivo_setor_up is not None:
            # Compara o conteúdo enviado com o que já está em disco para evitar
            # loop de rerun: o file_uploader devolve o mesmo objeto a cada execução.
            import hashlib
            novo_bytes = arquivo_setor_up.getbuffer().tobytes()
            novo_hash = hashlib.md5(novo_bytes).hexdigest()
            arquivo_disco = Path(ARQUIVO_SETOR_PADRAO)
            hash_disco = (hashlib.md5(arquivo_disco.read_bytes()).hexdigest()
                          if arquivo_disco.exists() else None)
            if novo_hash != hash_disco:
                arquivo_disco.write_bytes(novo_bytes)
                st.cache_data.clear()  # invalida cache do carregar_classificacao
                st.success("✅ Arquivo salvo! Recarregando...")
                st.rerun()
            # else: arquivo igual ao já salvo, não faz nada (evita loop)

        if classificacao_df is not None:
            n_setores = classificacao_df["SETOR"].nunique()
            n_acoes = classificacao_df.shape[0]
            st.success(f"✅ Arquivo ativo: **{n_acoes} ações** em **{n_setores} setores** mapeados.")
            if st.button("🗑️ Remover arquivo setorial"):
                Path(ARQUIVO_SETOR_PADRAO).unlink(missing_ok=True)
                st.cache_data.clear()
                st.rerun()

    if classificacao_df is None:
        st.stop()

    carteira_admin_atual = carregar_carteira_admin()

    col_input, col_lista = st.columns([1, 1])

    with col_input:
        st.markdown("#### ➕ Adicionar / atualizar lista")
        texto = st.text_area(
            "Tickers separados por vírgula",
            value=", ".join(carteira_admin_atual) if carteira_admin_atual else
                  "PETR4, VALE3, ITUB4, WEGE3, BBDC4, SUZB3, EQTL3, RADL3, RENT3, B3SA3, KLBN11, TOTS3, ABEV3",
            height=120,
            help="Idealmente cubra vários setores — um ou dois por setor."
        )

        if st.button("💾 Salvar carteira da casa", type="primary"):
            nova = [normalizar_ticker(t) for t in texto.split(",") if t.strip()]
            nova = list(dict.fromkeys(nova))  # dedup mantendo ordem
            salvar_carteira_admin(nova)
            st.success(f"✅ Salvas {len(nova)} ações.")
            st.rerun()

    with col_lista:
        st.markdown("#### 📋 Composição setorial atual da casa")
        if carteira_admin_atual:
            df_admin = pd.DataFrame({"Ticker": carteira_admin_atual})
            df_admin["Setor"] = df_admin["Ticker"].apply(
                lambda x: setor_de(x, classificacao_df)
            )
            cobertura = df_admin.groupby("Setor").size().reset_index(name="Qtd. ações")
            cobertura = cobertura.sort_values("Qtd. ações", ascending=False)

            st.dataframe(cobertura, width='stretch', hide_index=True)

            n_setores = cobertura.shape[0]
            n_setores_total = classificacao_df["SETOR"].nunique()
            st.metric("Cobertura setorial",
                      f"{n_setores}/{n_setores_total} setores",
                      help="Quantos setores a carteira da casa cobre")
        else:
            st.info("Nenhuma ação cadastrada ainda.")

    if carteira_admin_atual:
        st.markdown("#### 🔍 Detalhe das ações cadastradas")
        df_full = pd.DataFrame({"Ticker": carteira_admin_atual})
        df_full["Setor"] = df_full["Ticker"].apply(lambda x: setor_de(x, classificacao_df))
        st.dataframe(df_full, width='stretch', hide_index=True)

    st.stop()


# ============================================================
# MODO CLIENTE
# ============================================================
st.markdown("### 1️⃣ Conte sobre sua carteira")

# Inicializa estado para os pesos
if "carteira_calculada" not in st.session_state:
    st.session_state.carteira_calculada = False

with st.form("form_carteira"):
    tickers_input = st.text_area(
        "💼 Quais ações você tem? (separe por vírgula)",
        value="PETR4, VALE3, ITUB4, BBDC4, BBAS3",
        height=80,
        help="Digite os códigos das ações brasileiras que estão na sua carteira."
    )

    enviar = st.form_submit_button("🚀 Analisar minha carteira", type="primary",
                                    width='stretch')


if enviar:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("⚠️ Informe pelo menos 2 ações.")
        st.stop()

    if classificacao_df is None:
        st.error("📂 O sistema ainda não foi configurado pelo administrador. "
                 "Avise o gestor para concluir o cadastro.")
        st.stop()

    # Configurações vêm do admin (não do cliente)
    config_admin = carregar_config_admin()
    taxa_rf = float(config_admin.get("taxa_livre_risco", TAXA_RF_PADRAO))
    data_inicio = (pd.Timestamp.today() - pd.DateOffset(years=ANOS_HISTORICO)).strftime("%Y-%m-%d")

    # Junta tickers do cliente + tickers admin (para podermos sugerir substituições)
    # IMPORTANTE: usa normalizar_ticker (preserva número) — o Yahoo precisa de PETR4.SA, não PETR.SA
    carteira_admin = carregar_carteira_admin()
    tickers_universo = list(dict.fromkeys(
        [normalizar_ticker(t) for t in tickers] +
        [normalizar_ticker(t) for t in carteira_admin]
    ))

    with st.spinner("⏳ Baixando cotações dos últimos anos..."):
        try:
            carteira = CarteiraMarkowitz(tickers_universo, start=data_inicio)
            carteira.baixar_dados()
            carteira.calcular_metricas(taxa_livre_risco=taxa_rf)
        except Exception as e:
            st.error(f"❌ Erro: {e}")
            st.stop()

    st.session_state.carteira = carteira
    st.session_state.tickers_cliente_originais = [normalizar_ticker(t) for t in tickers]
    st.session_state.taxa_rf = taxa_rf
    st.session_state.carteira_calculada = True
    st.session_state.pesos_definidos = False


if st.session_state.carteira_calculada:
    carteira = st.session_state.carteira
    tickers_cliente = st.session_state.tickers_cliente_originais
    taxa_rf = st.session_state.taxa_rf

    # Filtra ativos que de fato baixaram
    ativos_cliente_validos = [a for a in tickers_cliente if a in carteira.nomes_ativos]
    ativos_admin = [a for a in carteira.nomes_ativos if a not in ativos_cliente_validos]

    if carteira.falhas_download:
        with st.expander("⚠️ Algumas ações não baixaram"):
            for t, erro in carteira.falhas_download:
                st.write(f"- **{t}**: {erro}")

    if len(ativos_cliente_validos) < 2:
        st.error("Não foi possível baixar dados de pelo menos 2 ações suas.")
        st.stop()

    st.markdown("---")
    st.markdown("### 2️⃣ Quanto você tem de cada ação?")
    st.caption("Informe o **percentual** que cada ação representa hoje na sua carteira. A soma precisa ser 100%.")

    n = len(ativos_cliente_validos)
    cols = st.columns(min(n, 5))
    pesos_cliente = []
    peso_default = round(1 / n, 4)

    for i, ativo in enumerate(ativos_cliente_validos):
        chave = f"peso_v2_{ativo}"
        if chave not in st.session_state:
            st.session_state[chave] = peso_default
        with cols[i % len(cols)]:
            p = st.number_input(
                f"**{ativo}**",
                min_value=0.0, max_value=1.0, step=0.01,
                format="%.4f", key=chave,
                help=f"Setor: {setor_de(ativo, classificacao_df)}"
            )
            pesos_cliente.append(p)

    soma = sum(pesos_cliente)

    col_soma1, col_soma2 = st.columns([3, 1])
    with col_soma1:
        if np.isclose(soma, 1.0, atol=1e-3):
            st.success(f"✅ Soma: {soma:.2%} — perfeito!")
        else:
            st.warning(f"⚠️ Soma atual: **{soma:.2%}** — ajuste para chegar a **100%**.")
    with col_soma2:
        if st.button("⚖️ Equalizar pesos", help="Distribui igualmente"):
            for ativo in ativos_cliente_validos:
                st.session_state[f"peso_v2_{ativo}"] = round(1 / n, 4)
            st.rerun()

    if not np.isclose(soma, 1.0, atol=1e-3):
        st.stop()

    # ============================================================
    # DIAGNÓSTICO DA CARTEIRA ATUAL
    # ============================================================
    st.markdown("---")
    st.markdown("### 3️⃣ Como está sua carteira hoje")

    stats_atual = carteira.estatisticas_carteira(
        # vetor completo do universo: zero para ações admin
        [pesos_cliente[ativos_cliente_validos.index(a)] if a in ativos_cliente_validos else 0
         for a in carteira.nomes_ativos],
        taxa_rf
    )

    # Métricas em cards grandes
    c1, c2, c3, c4 = st.columns(4)
    card_metric(c1, "📈 Retorno esperado/ano", pct(stats_atual["retorno_anual"]),
                help_text="Quanto sua carteira tende a render por ano, na média histórica.")
    card_metric(c2, "📉 Risco (volatilidade)", pct(stats_atual["volatilidade_anual"]),
                help_text="O quanto sua carteira oscila. Mais alto = mais sustos.")
    card_metric(c3, "⭐ Índice Sharpe", f'{stats_atual["sharpe"]:.2f}',
                help_text="Retorno por unidade de risco. Acima de 1 é bom; acima de 2 é excelente.")
    card_metric(c4, "📦 Nº de ações", str(sum(1 for p in pesos_cliente if p > 0.001)))

    # Gráficos pizza atual: ativos e setores
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(
            grafico_pizza_ativos(ativos_cliente_validos, pesos_cliente,
                                 "Distribuição por ação"),
            width='stretch'
        )
    with g2:
        st.plotly_chart(
            grafico_pizza_setor(ativos_cliente_validos, pesos_cliente,
                                classificacao_df, "Distribuição por setor"),
            width='stretch'
        )

    # Diagnóstico de problemas
    problemas = diagnosticar_carteira(ativos_cliente_validos, pesos_cliente, classificacao_df)

    st.markdown("#### 🔍 Diagnóstico")
    if not problemas:
        st.success("🎉 **Carteira saudável!** Todas as regras da casa estão sendo respeitadas.")
    else:
        n_alta = sum(1 for p in problemas if p["severidade"] == "alta")
        n_media = sum(1 for p in problemas if p["severidade"] == "media")
        msg = []
        if n_alta:
            msg.append(f"**{n_alta}** problema(s) sério(s)")
        if n_media:
            msg.append(f"**{n_media}** ponto(s) de atenção")
        st.error(f"Encontramos {' e '.join(msg)}:")

        for prob in problemas:
            cor = COR_RUIM if prob["severidade"] == "alta" else COR_ALERTA
            st.markdown(
                f"""<div style='background:#fef2f2;border-left:4px solid {cor};
                padding:12px 16px;border-radius:6px;margin:8px 0;'>
                <strong style='font-size:16px;'>{prob['icone']} {prob['titulo']}</strong><br>
                <span style='color:#475569'>{prob['descricao']}</span></div>""",
                unsafe_allow_html=True
            )

    # ============================================================
    # SUGESTÕES — 3 CARTEIRAS OTIMIZADAS
    # ============================================================
    st.markdown("---")
    st.markdown("### 4️⃣ Nossas sugestões para você")
    st.caption(
        f"As três sugestões respeitam as regras da casa: máx. **{LIMITE_POR_ATIVO:.0%}** por ativo "
        f"e máx. **{LIMITE_POR_SETOR:.0%}** por setor. "
        f"Quando algum setor seu está concentrado, completamos com ações da casa de **setores ausentes**."
    )

    # Decide o universo: ativos do cliente + admin de setores ausentes (se necessário)
    candidatos_subst, racional_subst = sugerir_substituicoes_setoriais(
        ativos_cliente_validos, pesos_cliente, ativos_admin, classificacao_df
    )

    # Universo de otimização: cliente + candidatos válidos
    universo = list(dict.fromkeys(ativos_cliente_validos + candidatos_subst))
    universo = [a for a in universo if a in carteira.nomes_ativos]

    # Reduz a carteira para esse universo (subindex)
    idx_universo = [carteira.nomes_ativos.index(a) for a in universo]
    nomes_ativos_full = carteira.nomes_ativos.copy()

    # Cria uma instância REAL de CarteiraMarkowitz para o sub-universo.
    # Isso garante que TODOS os métodos da classe (incluindo _verificar_factibilidade
    # e qualquer adição futura) estejam disponíveis, sem precisar listar manualmente.
    sub = CarteiraMarkowitz.__new__(CarteiraMarkowitz)
    sub.tickers_originais = universo
    sub.nomes_ativos = universo
    sub.precos = None
    sub.retornos = None
    sub.retorno_medio_anual = carteira.retorno_medio_anual.iloc[idx_universo]
    sub.cov_anual = carteira.cov_anual.iloc[idx_universo, idx_universo]
    sub.correlacao = (carteira.correlacao.iloc[idx_universo, idx_universo]
                      if carteira.correlacao is not None else None)
    sub.estatisticas_ativos = None
    sub.falhas_download = []
    sub.taxa_livre_risco_anual = taxa_rf
    sub.start = None
    sub.end = None

    mapa_setor_universo = montar_mapa_setor(universo, classificacao_df)

    if racional_subst:
        st.info(
            "🔄 **Substituições setoriais aplicadas** — adicionamos ações da carteira "
            "da casa para preencher setores que faltavam:\n\n" +
            "\n".join(f"- **{r['ticker']}** → setor *{r['setor_novo']}*" for r in racional_subst)
        )

    try:
        with st.spinner("🧮 Calculando as 3 melhores carteiras..."):
            sug_sharpe = sub.otimizar("max_sharpe", taxa_rf, LIMITE_POR_ATIVO,
                                       mapa_setor_universo, LIMITE_POR_SETOR)
            sug_segura = sub.otimizar("min_var", taxa_rf, LIMITE_POR_ATIVO,
                                       mapa_setor_universo, LIMITE_POR_SETOR)
            sug_retorno = sub.otimizar("max_retorno", taxa_rf, LIMITE_POR_ATIVO,
                                        mapa_setor_universo, LIMITE_POR_SETOR)
            fronteira = sub.fronteira_eficiente(35, LIMITE_POR_ATIVO,
                                                 mapa_setor_universo, LIMITE_POR_SETOR)
    except Exception as e:
        st.error(f"Não conseguimos otimizar com as regras da casa. Erro: {e}")
        st.stop()

    sugestoes = {
        "Mais Eficiência": sug_sharpe,
        "Mais Segurança": sug_segura,
        "Mais Retorno": sug_retorno,
    }

    # Cards comparativos das 3 sugestões + atual
    st.markdown("#### Compare suas opções")

    def linha_comparacao(nome, stats, cor_destaque, descricao):
        with st.container():
            st.markdown(
                f"""<div style='background:white;border-top:4px solid {cor_destaque};
                padding:16px 20px;border-radius:8px;
                box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:12px;'>
                <h4 style='margin:0 0 4px 0;color:{cor_destaque};'>{nome}</h4>
                <p style='color:#64748b;margin:0 0 12px 0;font-size:14px;'>{descricao}</p>
                </div>""", unsafe_allow_html=True
            )
            cc1, cc2, cc3, cc4 = st.columns(4)
            delta_ret = stats["retorno_anual"] - stats_atual["retorno_anual"]
            delta_risk = stats["volatilidade_anual"] - stats_atual["volatilidade_anual"]
            delta_sharpe = stats["sharpe"] - stats_atual["sharpe"]
            cc1.metric("Retorno/ano", pct(stats["retorno_anual"]),
                       delta=pct(delta_ret),
                       delta_color="normal" if delta_ret >= 0 else "inverse")
            cc2.metric("Risco", pct(stats["volatilidade_anual"]),
                       delta=pct(delta_risk),
                       delta_color="inverse" if delta_risk >= 0 else "normal")
            cc3.metric("Sharpe", f'{stats["sharpe"]:.2f}',
                       delta=f"{delta_sharpe:+.2f}",
                       delta_color="normal" if delta_sharpe >= 0 else "inverse")
            cc4.metric("Top ação",
                       universo[int(np.argmax(stats["pesos"]))],
                       delta=pct(np.max(stats["pesos"])))

    linha_comparacao(
        "⭐ Mais Eficiência (Maior Sharpe)", sug_sharpe, "#2563eb",
        "Melhor relação retorno/risco. A escolha 'equilibrada' — boa pra maioria dos perfis."
    )
    linha_comparacao(
        "🛡️ Mais Segurança (Menor Risco)", sug_segura, COR_OK,
        "Menor oscilação. Indicada se você prioriza dormir tranquilo a buscar ganhos altos."
    )
    linha_comparacao(
        "🚀 Mais Retorno", sug_retorno, "#9333ea",
        "Maior retorno esperado. Perfil mais agressivo — com risco proporcionalmente maior."
    )

    # Gráfico Risco x Retorno com as opções marcadas
    st.markdown("#### 🗺️ Mapa de risco × retorno")
    st.caption("Cada ponto colorido é uma sugestão. O 'X' vermelho é onde sua carteira está hoje.")
    st.plotly_chart(grafico_risco_retorno(stats_atual, sugestoes, fronteira),
                    width='stretch')

    # ============================================================
    # DETALHES DE CADA SUGESTÃO — abas
    # ============================================================
    st.markdown("#### 📑 Detalhe de cada sugestão")

    aba1, aba2, aba3 = st.tabs([
        "⭐ Mais Eficiência", "🛡️ Mais Segurança", "🚀 Mais Retorno"
    ])

    def renderizar_aba(stats, nome_curto):
        # Pesos sugeridos no universo completo (cliente + candidatos)
        pesos_sug = stats["pesos"]

        # Vetor de pesos atual estendido para o universo (zeros para admin)
        pesos_atual_universo = []
        for a in universo:
            if a in ativos_cliente_validos:
                pesos_atual_universo.append(
                    pesos_cliente[ativos_cliente_validos.index(a)]
                )
            else:
                pesos_atual_universo.append(0.0)

        # Tabela de mudanças
        df_mud = tabela_mudancas(universo, pesos_atual_universo, pesos_sug, classificacao_df)

        col_pizza, col_tabela = st.columns([1, 1.3])
        with col_pizza:
            st.plotly_chart(
                grafico_pizza_ativos(universo, pesos_sug,
                                     f"Composição — {nome_curto}"),
                width='stretch',
                key=f"pizza_{nome_curto}"
            )

        with col_tabela:
            st.markdown("**Mudanças sugeridas**")
            df_show = df_mud.copy()
            df_show["Hoje"] = df_show["Hoje"].apply(lambda x: f"{x:.1%}")
            df_show["Sugerido"] = df_show["Sugerido"].apply(lambda x: f"{x:.1%}")
            df_show["Mudança"] = df_show["Mudança"].apply(lambda x: f"{x:+.1%}")
            st.dataframe(df_show, width='stretch', hide_index=True, height=380)

        # Resumo: o que mudou
        comprar = df_mud[df_mud["Ação"] == "🟢 COMPRAR"]
        vender = df_mud[df_mud["Ação"] == "🔴 VENDER"]
        aumentar = df_mud[df_mud["Ação"] == "↑ Aumentar"]
        reduzir = df_mud[df_mud["Ação"] == "↓ Reduzir"]

        cR, cG = st.columns(2)
        with cR:
            st.markdown("**🟢 Para entrar / aumentar**")
            if len(comprar) + len(aumentar) == 0:
                st.write("_(nada a aumentar)_")
            else:
                for _, r in comprar.iterrows():
                    st.markdown(f"- **COMPRAR {r['Ativo']}** ({r['Setor']}) → "
                                f"alocar **{r['Sugerido']:.1%}**")
                for _, r in aumentar.iterrows():
                    st.markdown(f"- **{r['Ativo']}**: {r['Hoje']:.1%} → "
                                f"**{r['Sugerido']:.1%}** ({r['Mudança']:+.1%})")
        with cG:
            st.markdown("**🔴 Para sair / reduzir**")
            if len(vender) + len(reduzir) == 0:
                st.write("_(nada a reduzir)_")
            else:
                for _, r in vender.iterrows():
                    st.markdown(f"- **VENDER {r['Ativo']}** ({r['Setor']})")
                for _, r in reduzir.iterrows():
                    st.markdown(f"- **{r['Ativo']}**: {r['Hoje']:.1%} → "
                                f"**{r['Sugerido']:.1%}** ({r['Mudança']:+.1%})")

        # Comparativo setorial
        st.plotly_chart(
            grafico_barras_setor_com_limite(
                universo, pesos_atual_universo,
                universo, pesos_sug, classificacao_df,
                titulo=f"Setores: atual × {nome_curto}"
            ),
            width='stretch',
            key=f"setor_{nome_curto}"
        )

    with aba1:
        renderizar_aba(sug_sharpe, "Mais Eficiência")
    with aba2:
        renderizar_aba(sug_segura, "Mais Segurança")
    with aba3:
        renderizar_aba(sug_retorno, "Mais Retorno")

    # ============================================================
    # RESUMO FINAL
    # ============================================================
    st.markdown("---")
    st.markdown("### 5️⃣ Resumo executivo")

    melhor_para_perfil = {
        "Conservador 🛡️": "Mais Segurança",
        "Moderado ⚖️": "Mais Eficiência",
        "Arrojado 🚀": "Mais Retorno"
    }

    st.markdown(
        f"""
- **Sua carteira hoje** rende {pct(stats_atual['retorno_anual'])} ao ano com risco de
  {pct(stats_atual['volatilidade_anual'])} (Sharpe {stats_atual['sharpe']:.2f}).
- Encontramos **{len(problemas)} ponto(s)** que merecem atenção.
- Comparando com nossas 3 sugestões:
  - **Conservador** → escolha *Mais Segurança*: risco
    {pct(sug_segura['volatilidade_anual'])} (vs. {pct(stats_atual['volatilidade_anual'])} hoje).
  - **Moderado** → escolha *Mais Eficiência*: Sharpe **{sug_sharpe['sharpe']:.2f}**
    (vs. {stats_atual['sharpe']:.2f} hoje).
  - **Arrojado** → escolha *Mais Retorno*: retorno {pct(sug_retorno['retorno_anual'])}
    ao ano.
        """
    )

    st.caption(
        "ℹ️ Esta análise é baseada em dados históricos e na teoria de Markowitz. "
        "Retornos passados não garantem resultados futuros. Considere conversar com seu assessor."
    )

else:
    st.info("👆 Preencha sua carteira e clique em **Analisar minha carteira** para começar.")

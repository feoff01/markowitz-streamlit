import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from pathlib import Path

TRADING_DAYS = 252
LIMITE_SETOR = 0.40
ARQUIVO_SETOR_PADRAO = "ClassifSetorial.xlsx"


class CarteiraMarkowitz:
    def __init__(self, tickers, start="2018-01-01", end=None):
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
        series = []
        falhas = []

        for ticker in self.tickers_originais:
            try:
                s = self.baixar_um_ticker(ticker)
                series.append(s)
            except Exception as e:
                falhas.append((ticker, str(e)))

        self.falhas_download = falhas

        if not series:
            raise ValueError(
                "Nenhum ticker retornou dados. Verifique internet, período, tickers ou bloqueio temporário do Yahoo."
            )

        precos = pd.concat(series, axis=1).sort_index()
        precos.index = pd.to_datetime(precos.index)
        precos = precos.dropna(how="all")
        precos = precos.dropna(axis=1, thresh=max(30, int(len(precos) * 0.7)))
        precos = precos.ffill().dropna()

        if precos.shape[1] < 2:
            raise ValueError("Restaram menos de 2 ativos válidos após limpeza dos dados.")

        self.precos = precos
        self.nomes_ativos = list(precos.columns)
        return self.precos

    def calcular_metricas(self, taxa_livre_risco=0.0):
        retornos = self.precos.pct_change().dropna()

        if retornos.empty:
            raise ValueError("Não foi possível calcular retornos.")

        retorno_medio_diario = retornos.mean()
        cov_diaria = retornos.cov()

        self.retornos = retornos
        self.retorno_medio_anual = retorno_medio_diario * TRADING_DAYS
        self.cov_anual = cov_diaria * TRADING_DAYS
        self.correlacao = retornos.corr()
        self.taxa_livre_risco_anual = float(taxa_livre_risco)

        volatilidade_anual = retornos.std() * np.sqrt(TRADING_DAYS)
        retorno_excesso_anual = self.retorno_medio_anual - self.taxa_livre_risco_anual
        sharpe_individual = np.where(
            volatilidade_anual > 0,
            retorno_excesso_anual / volatilidade_anual,
            np.nan
        )

        self.estatisticas_ativos = pd.DataFrame({
            "retorno_anual": self.retorno_medio_anual,
            "volatilidade_anual": volatilidade_anual,
            "sharpe": sharpe_individual
        }).sort_values("sharpe", ascending=False)

    def validar_pesos(self, pesos):
        pesos = np.array(pesos, dtype=float)
        if len(pesos) != len(self.nomes_ativos):
            raise ValueError("Quantidade de pesos diferente da quantidade de ativos.")
        if not np.isclose(pesos.sum(), 1.0, atol=1e-6):
            raise ValueError(f"A soma dos pesos deve ser 1. Soma atual: {pesos.sum():.6f}")
        return pesos

    def estatisticas_carteira(self, pesos, taxa_livre_risco=0.0):
        pesos = self.validar_pesos(pesos)
        retorno = float(np.dot(pesos, self.retorno_medio_anual.values))
        volatilidade = float(np.sqrt(np.dot(pesos.T, np.dot(self.cov_anual.values, pesos))))
        sharpe = (retorno - taxa_livre_risco) / volatilidade if volatilidade > 0 else np.nan

        return {
            "pesos": pesos,
            "retorno_anual": retorno,
            "volatilidade_anual": volatilidade,
            "risco_anual": volatilidade,
            "sharpe": sharpe
        }

    def mostrar_pesos(self, pesos):
        return pd.Series(np.array(pesos), index=self.nomes_ativos).sort_values(ascending=False)

    def _montar_restricoes(self, retorno_alvo=None, mapa_setor=None, limite_setor=None):
        restricoes = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if retorno_alvo is not None:
            restricoes.append(
                {"type": "eq", "fun": lambda w, r=retorno_alvo: np.dot(w, self.retorno_medio_anual.values) - r}
            )

        if mapa_setor and limite_setor is not None:
            setores = pd.Series(mapa_setor).reindex(self.nomes_ativos).fillna("Sem setor")
            for setor in setores.unique():
                idx = np.where(setores.values == setor)[0]
                restricoes.append(
                    {"type": "ineq", "fun": lambda w, idx=idx, lim=limite_setor: lim - np.sum(w[idx])}
                )

        return restricoes

    def otimizar_minima_variancia(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0,
                                  mapa_setor=None, limite_setor=None):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = self._montar_restricoes(mapa_setor=mapa_setor, limite_setor=limite_setor)

        resultado = minimize(
            fun=lambda w: np.dot(w.T, np.dot(self.cov_anual.values, w)),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if not resultado.success:
            raise RuntimeError(resultado.message)
        return self.estatisticas_carteira(resultado.x, taxa_livre_risco)

    def otimizar_max_sharpe(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0,
                            mapa_setor=None, limite_setor=None):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = self._montar_restricoes(mapa_setor=mapa_setor, limite_setor=limite_setor)

        def objetivo(w):
            retorno = np.dot(w, self.retorno_medio_anual.values)
            vol = np.sqrt(np.dot(w.T, np.dot(self.cov_anual.values, w)))
            if vol <= 0:
                return 1e10
            return -((retorno - taxa_livre_risco) / vol)

        resultado = minimize(
            fun=objetivo,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if not resultado.success:
            raise RuntimeError(resultado.message)
        return self.estatisticas_carteira(resultado.x, taxa_livre_risco)

    def otimizar_max_retorno(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0,
                             mapa_setor=None, limite_setor=None):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = self._montar_restricoes(mapa_setor=mapa_setor, limite_setor=limite_setor)

        resultado = minimize(
            fun=lambda w: -np.dot(w, self.retorno_medio_anual.values),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        if not resultado.success:
            raise RuntimeError(resultado.message)
        return self.estatisticas_carteira(resultado.x, taxa_livre_risco)

    def fronteira_eficiente(self, n_pontos=50, permitir_short=False, peso_maximo=1.0,
                            mapa_setor=None, limite_setor=None, taxa_livre_risco=0.0,
                            *args, **kwargs):
        # Compatibilidade extra: se alguma versão antiga chamar com argumentos
        # posicionais/nominais diferentes, preserva a execução.
        if "taxa_livre_risco" in kwargs and kwargs["taxa_livre_risco"] is not None:
            taxa_livre_risco = kwargs["taxa_livre_risco"]
        if "mapa_setor" in kwargs and kwargs["mapa_setor"] is not None:
            mapa_setor = kwargs["mapa_setor"]
        if "limite_setor" in kwargs and kwargs["limite_setor"] is not None:
            limite_setor = kwargs["limite_setor"]

        min_var = self.otimizar_minima_variancia(
            taxa_livre_risco=taxa_livre_risco,
            permitir_short=permitir_short,
            peso_maximo=peso_maximo,
            mapa_setor=mapa_setor,
            limite_setor=limite_setor,
        )
        max_ret = self.otimizar_max_retorno(
            taxa_livre_risco=taxa_livre_risco,
            permitir_short=permitir_short,
            peso_maximo=peso_maximo,
            mapa_setor=mapa_setor,
            limite_setor=limite_setor,
        )

        retorno_min = float(min_var["retorno_anual"])
        retorno_max = float(max_ret["retorno_anual"])
        if retorno_max <= retorno_min + 1e-10:
            retorno_max = retorno_min + 1e-6

        retornos_alvo = np.linspace(retorno_min, retorno_max, n_pontos)
        n = len(self.nomes_ativos)
        x0 = np.array(min_var["pesos"], dtype=float)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))

        resultados = []

        for retorno_alvo in retornos_alvo:
            constraints = self._montar_restricoes(
                retorno_alvo=retorno_alvo,
                mapa_setor=mapa_setor,
                limite_setor=limite_setor
            )

            resultado = minimize(
                fun=lambda w: np.dot(w.T, np.dot(self.cov_anual.values, w)),
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500}
            )

            if resultado.success:
                stats = self.estatisticas_carteira(resultado.x, taxa_livre_risco)
                resultados.append({
                    "risco": stats["risco_anual"],
                    "volatilidade": stats["volatilidade_anual"],
                    "retorno": stats["retorno_anual"],
                    "sharpe": stats["sharpe"]
                })
                x0 = np.array(resultado.x, dtype=float)

        return pd.DataFrame(resultados).drop_duplicates(subset=["risco", "retorno"]).sort_values("risco")

    def gerar_carteiras_aleatorias(self, numero_carteiras=3000, taxa_livre_risco=0.0,
                                   mapa_setor=None, limite_setor=None):
        n = len(self.nomes_ativos)
        resultados = []
        setores = pd.Series(mapa_setor).reindex(self.nomes_ativos).fillna("Sem setor") if mapa_setor else None

        for _ in range(numero_carteiras):
            pesos = np.random.random(n)
            pesos /= pesos.sum()

            if limite_setor is not None and setores is not None:
                expo = pd.Series(pesos, index=self.nomes_ativos).groupby(setores).sum()
                if (expo > limite_setor + 1e-8).any():
                    continue

            retorno = float(np.dot(pesos, self.retorno_medio_anual.values))
            vol = float(np.sqrt(np.dot(pesos.T, np.dot(self.cov_anual.values, pesos))))
            sharpe = (retorno - taxa_livre_risco) / vol if vol > 0 else np.nan

            resultados.append({
                "retorno": retorno,
                "volatilidade": vol,
                "risco": vol,
                "sharpe": sharpe
            })

        return pd.DataFrame(resultados)


def pct(x):
    return f"{x:.2%}"

def normalizar_codigo_setorial(codigo):
    codigo = str(codigo).upper().strip().replace(".SA", "")
    letras = "".join(ch for ch in codigo if ch.isalpha())
    return letras if letras else codigo


def carregar_classificacao_setorial(arquivo_excel):
    df = pd.read_excel(arquivo_excel)
    df = df.rename(columns={df.columns[0]: "SETOR", df.columns[1]: "SUBSETOR", df.columns[2]: "CODIGO"})
    df["CODIGO"] = df["CODIGO"].apply(normalizar_codigo_setorial)
    df["SETOR"] = df["SETOR"].fillna("Sem setor")
    df["SUBSETOR"] = df["SUBSETOR"].fillna("Sem subsetor")
    df = df[df["CODIGO"].notna()]
    df = df[df["CODIGO"] != "CÓDIGO"]
    df = df.drop_duplicates(subset=["CODIGO"])
    return df


def montar_info_setorial(ativos, classificacao_df):
    base = pd.DataFrame({"ativo": ativos})
    base["CODIGO_REF"] = base["ativo"].apply(normalizar_codigo_setorial)
    base = base.merge(classificacao_df, left_on="CODIGO_REF", right_on="CODIGO", how="left")
    base["SETOR"] = base["SETOR"].fillna("Sem setor")
    base["SUBSETOR"] = base["SUBSETOR"].fillna("Sem subsetor")
    return base[["ativo", "SETOR", "SUBSETOR"]]


def analisar_concentracao_setorial(ativos, pesos, classificacao_df, limite_setor=LIMITE_SETOR):
    df = pd.DataFrame({"ativo": ativos, "peso": pesos})
    df["CODIGO_REF"] = df["ativo"].apply(normalizar_codigo_setorial)
    df = df.merge(classificacao_df, left_on="CODIGO_REF", right_on="CODIGO", how="left")
    df["SETOR"] = df["SETOR"].fillna("Sem setor")
    df["SUBSETOR"] = df["SUBSETOR"].fillna("Sem subsetor")

    exposicao_setorial = (
        df.groupby("SETOR", dropna=False)["peso"]
        .sum()
        .sort_values(ascending=False)
        .rename("peso_setor")
        .reset_index()
    )
    exposicao_setorial["acima_limite"] = exposicao_setorial["peso_setor"] > limite_setor + 1e-8

    return df, exposicao_setorial


def gerar_tabela_recomendacao(carteira, pesos_atuais, pesos_recomendados, classificacao_df):
    atual = pd.Series(pesos_atuais, index=carteira.nomes_ativos, name="peso_atual")
    rec = pd.Series(pesos_recomendados, index=carteira.nomes_ativos, name="peso_recomendado")
    sharpe = carteira.estatisticas_ativos["sharpe"].reindex(carteira.nomes_ativos)

    df = pd.concat([atual, rec, sharpe], axis=1).reset_index().rename(columns={"index": "ativo"})
    df["CODIGO_REF"] = df["ativo"].apply(normalizar_codigo_setorial)
    df = df.merge(classificacao_df, left_on="CODIGO_REF", right_on="CODIGO", how="left")
    df["SETOR"] = df["SETOR"].fillna("Sem setor")
    df["SUBSETOR"] = df["SUBSETOR"].fillna("Sem subsetor")
    df["delta"] = df["peso_recomendado"] - df["peso_atual"]

    def acao(delta):
        if delta > 0.005:
            return "Aumentar"
        if delta < -0.005:
            return "Reduzir"
        return "Manter"

    df["ação"] = df["delta"].apply(acao)
    df = df.sort_values(["ação", "delta", "sharpe"], ascending=[True, True, False])
    return df[["ativo", "SETOR", "SUBSETOR", "sharpe", "peso_atual", "peso_recomendado", "delta", "ação"]]


def gerar_resumo_textual(df_recomendacao, exposicao_atual, exposicao_recomendada, limite_setor=LIMITE_SETOR):
    mensagens = []

    setores_acima = exposicao_atual[exposicao_atual["peso_setor"] > limite_setor + 1e-8]
    if not setores_acima.empty:
        texto = ", ".join([f"{row.SETOR} ({row.peso_setor:.1%})" for row in setores_acima.itertuples()])
        mensagens.append(f"Há concentração setorial acima do limite de {limite_setor:.0%} em: {texto}.")
    else:
        mensagens.append(f"A carteira não ultrapassa o limite de {limite_setor:.0%} por setor.")

    top_reduce = df_recomendacao[df_recomendacao["ação"] == "Reduzir"].sort_values("delta").head(3)
    top_add = df_recomendacao[df_recomendacao["ação"] == "Aumentar"].sort_values("delta", ascending=False).head(3)

    if not top_reduce.empty:
        msgs = [f"{r.ativo} ({r.delta:.1%})" for r in top_reduce.itertuples()]
        mensagens.append("Ativos para reduzir: " + ", ".join(msgs) + ".")

    if not top_add.empty:
        msgs = [f"{r.ativo} (+{r.delta:.1%})" for r in top_add.itertuples()]
        mensagens.append("Ativos para aumentar: " + ", ".join(msgs) + ".")

    top_setores_rec = exposicao_recomendada.sort_values("peso_setor", ascending=False).head(3)
    if not top_setores_rec.empty:
        msgs = [f"{r.SETOR} ({r.peso_setor:.1%})" for r in top_setores_rec.itertuples()]
        mensagens.append("Após a realocação sugerida, os maiores pesos setoriais ficam em: " + ", ".join(msgs) + ".")

    return mensagens


st.set_page_config(page_title="Otimizador Markowitz + Setorial", layout="wide")
st.title("Otimizador de Carteira: Markowitz + Concentração Setorial")
st.caption("Combina análise quantitativa de risco/retorno com limites de concentração por setor.")

if "dados_calculados" not in st.session_state:
    st.session_state.dados_calculados = False

with st.sidebar:
    st.header("Configurações")

    tickers_input = st.text_area(
        "Tickers separados por vírgula",
        value="PETR4, VALE3, ITUB4, WEGE3, BBDC4, PRIO3"
    )

    data_inicio = st.date_input("Data inicial", value=pd.to_datetime("2018-01-01"))
    data_fim = st.date_input("Data final", value=pd.to_datetime("today"))
    taxa_livre_risco = st.number_input("Taxa livre de risco anual", value=0.10, step=0.01)
    permitir_short = st.checkbox("Permitir short", value=False)
    peso_maximo = st.slider("Peso máximo por ativo", 0.05, 1.0, 0.35, 0.05)
    limite_setor = st.slider("Limite máximo por setor", 0.10, 1.0, 0.40, 0.05)
    numero_carteiras = st.slider("Carteiras aleatórias", 500, 10000, 3000, 500)
    usar_restricao_setorial = st.checkbox("Aplicar limite setorial na otimização", value=True)

    st.markdown("---")
    st.subheader("Classificação setorial")
    arquivo_setor = st.file_uploader(
        "Envie o Excel de setores (opcional)",
        type=["xlsx", "xls"],
        help="Se não enviar, o app tenta usar o arquivo ClassifSetorial.xlsx da pasta local."
    )

    calcular = st.button("Calcular / Atualizar dados")

if calcular:
    try:
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Digite pelo menos 2 tickers.")
            st.stop()

        caminho_excel = None
        if arquivo_setor is not None:
            caminho_excel = arquivo_setor
        else:
            caminho_padrao = Path(ARQUIVO_SETOR_PADRAO)
            if caminho_padrao.exists():
                caminho_excel = str(caminho_padrao)

        with st.spinner("Baixando dados e calculando..."):
            carteira = CarteiraMarkowitz(tickers, start=str(data_inicio), end=str(data_fim))
            carteira.baixar_dados()
            carteira.calcular_metricas(taxa_livre_risco=taxa_livre_risco)

            classificacao_df = None
            info_setorial = None
            mapa_setor = None
            if caminho_excel is not None:
                classificacao_df = carregar_classificacao_setorial(caminho_excel)
                info_setorial = montar_info_setorial(carteira.nomes_ativos, classificacao_df)
                mapa_setor = info_setorial.set_index("ativo")["SETOR"].to_dict()

            st.session_state.carteira = carteira
            st.session_state.classificacao_df = classificacao_df
            st.session_state.info_setorial = info_setorial
            st.session_state.mapa_setor = mapa_setor
            st.session_state.taxa_livre_risco = taxa_livre_risco
            st.session_state.permitir_short = permitir_short
            st.session_state.peso_maximo = peso_maximo
            st.session_state.numero_carteiras = numero_carteiras
            st.session_state.limite_setor = limite_setor
            st.session_state.usar_restricao_setorial = usar_restricao_setorial
            st.session_state.dados_calculados = True

    except Exception as e:
        st.exception(e)

if st.session_state.dados_calculados:
    carteira = st.session_state.carteira
    classificacao_df = st.session_state.classificacao_df
    info_setorial = st.session_state.info_setorial
    mapa_setor = st.session_state.mapa_setor
    taxa_livre_risco = st.session_state.taxa_livre_risco
    permitir_short = st.session_state.permitir_short
    peso_maximo = st.session_state.peso_maximo
    numero_carteiras = st.session_state.numero_carteiras
    limite_setor = st.session_state.limite_setor
    usar_restricao_setorial = st.session_state.usar_restricao_setorial

    mapa_setor_otimizacao = mapa_setor if usar_restricao_setorial else None
    limite_setor_otimizacao = limite_setor if usar_restricao_setorial else None

    if carteira.falhas_download:
        with st.expander("Tickers com falha no download"):
            for ticker, erro in carteira.falhas_download:
                st.write(f"{ticker}: {erro}")

    ativos = carteira.nomes_ativos
    n = len(ativos)

    st.subheader("Ativos válidos")
    st.write(", ".join(ativos))

    if info_setorial is not None:
        st.subheader("Classificação das ações escolhidas")
        st.dataframe(info_setorial, use_container_width=True)
    else:
        st.warning("Nenhum arquivo setorial foi encontrado. A análise setorial ficará desabilitada.")

    st.subheader("Estatísticas dos ativos")
    st.caption(f"Sharpe padronizado: (retorno anual - taxa livre de risco) / volatilidade anual | rf = {taxa_livre_risco:.2%}")
    st.dataframe(
        carteira.estatisticas_ativos.style.format({
            "retorno_anual": "{:.2%}",
            "volatilidade_anual": "{:.2%}",
            "sharpe": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Correlação")
    fig_corr = px.imshow(carteira.correlacao, text_auto=".2f", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Pesos da sua carteira")
    st.caption("Preencha os pesos da carteira atual do investidor. A soma precisa ser 100%. Se quiser, use o botão para normalizar automaticamente.")
    cols = st.columns(min(n, 6))
    pesos_usuario = []

    for i, ativo in enumerate(ativos):
        chave = f"peso_{ativo}"
        if chave not in st.session_state:
            st.session_state[chave] = round(1 / n, 4)

        with cols[i % len(cols)]:
            p = st.number_input(
                ativo,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=chave
            )
            pesos_usuario.append(p)

    soma = sum(pesos_usuario)
    csum1, csum2 = st.columns([1, 1])
    csum1.write(f"Soma dos pesos: {soma:.4f}")
    if csum2.button("Normalizar pesos automaticamente"):
        if soma > 0:
            for ativo, p in zip(ativos, pesos_usuario):
                st.session_state[f"peso_{ativo}"] = float(p / soma)
            st.rerun()

    stats_usuario = None
    exposicao_setorial_atual = None
    exposicao_setorial_recomendada = None
    recomendacao_df = None

    if np.isclose(soma, 1.0, atol=1e-6):
        stats_usuario = carteira.estatisticas_carteira(pesos_usuario, taxa_livre_risco)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno da sua carteira", pct(stats_usuario["retorno_anual"]))
        c2.metric("Volatilidade da sua carteira", pct(stats_usuario["volatilidade_anual"]))
        c3.metric("Risco da sua carteira", pct(stats_usuario["risco_anual"]))
        c4.metric("Sharpe da sua carteira", f'{stats_usuario["sharpe"]:.4f}')

        st.dataframe(
            carteira.mostrar_pesos(stats_usuario["pesos"]).to_frame("peso").style.format("{:.2%}"),
            use_container_width=True
        )

        if classificacao_df is not None:
            _, exposicao_setorial_atual = analisar_concentracao_setorial(
                ativos, pesos_usuario, classificacao_df, limite_setor=limite_setor
            )
            st.subheader("Concentração setorial da carteira atual")
            st.dataframe(
                exposicao_setorial_atual.style.format({"peso_setor": "{:.2%}"}),
                use_container_width=True
            )

            fig_setor_atual = px.pie(
                exposicao_setorial_atual,
                names="SETOR",
                values="peso_setor",
                title="Distribuição setorial da carteira atual"
            )
            st.plotly_chart(fig_setor_atual, use_container_width=True)
    else:
        st.warning("A soma dos pesos deve ser 1 para calcular a sua carteira.")

    min_var = carteira.otimizar_minima_variancia(
        taxa_livre_risco, permitir_short, peso_maximo, mapa_setor_otimizacao, limite_setor_otimizacao
    )
    max_sharpe = carteira.otimizar_max_sharpe(
        taxa_livre_risco, permitir_short, peso_maximo, mapa_setor_otimizacao, limite_setor_otimizacao
    )
    max_retorno = carteira.otimizar_max_retorno(
        taxa_livre_risco, permitir_short, peso_maximo, mapa_setor_otimizacao, limite_setor_otimizacao
    )

    st.subheader("Carteiras ótimas")
    aba1, aba2, aba3 = st.tabs(["Menor risco", "Maior Sharpe", "Maior retorno"])

    with aba1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno", pct(min_var["retorno_anual"]))
        c2.metric("Volatilidade", pct(min_var["volatilidade_anual"]))
        c3.metric("Risco", pct(min_var["risco_anual"]))
        c4.metric("Sharpe", f'{min_var["sharpe"]:.4f}')
        st.dataframe(
            carteira.mostrar_pesos(min_var["pesos"]).to_frame("peso").style.format("{:.2%}"),
            use_container_width=True
        )

    with aba2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno", pct(max_sharpe["retorno_anual"]))
        c2.metric("Volatilidade", pct(max_sharpe["volatilidade_anual"]))
        c3.metric("Risco", pct(max_sharpe["risco_anual"]))
        c4.metric("Sharpe", f'{max_sharpe["sharpe"]:.4f}')
        st.dataframe(
            carteira.mostrar_pesos(max_sharpe["pesos"]).to_frame("peso").style.format("{:.2%}"),
            use_container_width=True
        )

    with aba3:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retorno", pct(max_retorno["retorno_anual"]))
        c2.metric("Volatilidade", pct(max_retorno["volatilidade_anual"]))
        c3.metric("Risco", pct(max_retorno["risco_anual"]))
        c4.metric("Sharpe", f'{max_retorno["sharpe"]:.4f}')
        st.dataframe(
            carteira.mostrar_pesos(max_retorno["pesos"]).to_frame("peso").style.format("{:.2%}"),
            use_container_width=True
        )

    st.subheader("Fronteira eficiente")
    fronteira = carteira.fronteira_eficiente(
        n_pontos=50,
        permitir_short=permitir_short,
        peso_maximo=peso_maximo,
        mapa_setor=mapa_setor_otimizacao,
        limite_setor=limite_setor_otimizacao,
        taxa_livre_risco=taxa_livre_risco,
    )
    aleatorias = carteira.gerar_carteiras_aleatorias(
        numero_carteiras, taxa_livre_risco, mapa_setor_otimizacao, limite_setor_otimizacao
    )

    fig = go.Figure()

    if not aleatorias.empty:
        fig.add_trace(go.Scatter(
            x=aleatorias["risco"],
            y=aleatorias["retorno"],
            mode="markers",
            marker=dict(size=5, color=aleatorias["sharpe"], colorscale="Viridis", showscale=True),
            name="Carteiras aleatórias"
        ))

    if not fronteira.empty:
        fig.add_trace(go.Scatter(
            x=fronteira["risco"],
            y=fronteira["retorno"],
            mode="lines",
            name="Fronteira eficiente"
        ))
    else:
        st.warning("Não foi possível traçar a fronteira eficiente completa com as restrições atuais. O gráfico abaixo mostra as carteiras viáveis encontradas e os pontos ótimos.")

    fig.add_trace(go.Scatter(
        x=[min_var["risco_anual"]],
        y=[min_var["retorno_anual"]],
        mode="markers",
        name="Menor risco",
        marker=dict(size=12)
    ))

    fig.add_trace(go.Scatter(
        x=[max_sharpe["risco_anual"]],
        y=[max_sharpe["retorno_anual"]],
        mode="markers",
        name="Maior Sharpe",
        marker=dict(size=14, symbol="star")
    ))

    fig.add_trace(go.Scatter(
        x=[max_retorno["risco_anual"]],
        y=[max_retorno["retorno_anual"]],
        mode="markers",
        name="Maior retorno",
        marker=dict(size=12, symbol="diamond")
    ))

    if stats_usuario is not None:
        fig.add_trace(go.Scatter(
            x=[stats_usuario["risco_anual"]],
            y=[stats_usuario["retorno_anual"]],
            mode="markers",
            name="Sua carteira",
            marker=dict(size=12, symbol="x")
        ))

    fig.update_layout(
        title="Risco x Retorno",
        xaxis_title="Risco anual",
        yaxis_title="Retorno anual",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Carteira final recomendada")
    st.caption("Esta é a carteira de maior Sharpe dentro das restrições definidas. Se a restrição setorial estiver ligada, ela respeita o limite máximo por setor.")

    pesos_finais_df = carteira.mostrar_pesos(max_sharpe["pesos"]).to_frame("peso_recomendado")
    st.dataframe(
        pesos_finais_df.style.format({"peso_recomendado": "{:.2%}"}),
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno esperado", pct(max_sharpe["retorno_anual"]))
    c2.metric("Volatilidade esperada", pct(max_sharpe["volatilidade_anual"]))
    c3.metric("Sharpe esperado", f'{max_sharpe["sharpe"]:.4f}')
    c4.metric("Maior peso individual", pct(np.max(max_sharpe["pesos"])))

    if stats_usuario is not None:
        st.subheader("Recomendação de realocação")
        st.caption("A recomendação abaixo compara a carteira atual com a carteira final recomendada de maior Sharpe.")

        recomendacao_df = gerar_tabela_recomendacao(
            carteira=carteira,
            pesos_atuais=stats_usuario["pesos"],
            pesos_recomendados=max_sharpe["pesos"],
            classificacao_df=classificacao_df if classificacao_df is not None else pd.DataFrame(columns=["CODIGO", "SETOR", "SUBSETOR"])
        )

        st.dataframe(
            recomendacao_df.style.format({
                "sharpe": "{:.4f}",
                "peso_atual": "{:.2%}",
                "peso_recomendado": "{:.2%}",
                "delta": "{:+.2%}"
            }),
            use_container_width=True
        )

        if classificacao_df is not None:
            _, exposicao_setorial_recomendada = analisar_concentracao_setorial(
                ativos, max_sharpe["pesos"], classificacao_df, limite_setor=limite_setor
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Setores na carteira atual**")
                st.dataframe(
                    exposicao_setorial_atual.style.format({"peso_setor": "{:.2%}"}) if exposicao_setorial_atual is not None else exposicao_setorial_recomendada.style.format({"peso_setor": "{:.2%}"}),
                    use_container_width=True
                )

            with c2:
                st.markdown("**Setores na carteira recomendada**")
                st.dataframe(
                    exposicao_setorial_recomendada.style.format({"peso_setor": "{:.2%}"}),
                    use_container_width=True
                )

            resumo = gerar_resumo_textual(
                recomendacao_df,
                exposicao_setorial_atual if exposicao_setorial_atual is not None else exposicao_setorial_recomendada,
                exposicao_setorial_recomendada,
                limite_setor=limite_setor
            )

            st.markdown("### Resumo executivo")
            for msg in resumo:
                st.info(msg)

            fig_comp = go.Figure()
            atual_plot = exposicao_setorial_atual if exposicao_setorial_atual is not None else exposicao_setorial_recomendada.copy()
            fig_comp.add_trace(go.Bar(
                x=atual_plot["SETOR"],
                y=atual_plot["peso_setor"],
                name="Atual"
            ))
            fig_comp.add_trace(go.Bar(
                x=exposicao_setorial_recomendada["SETOR"],
                y=exposicao_setorial_recomendada["peso_setor"],
                name="Recomendada"
            ))
            fig_comp.add_hline(y=limite_setor, line_dash="dash", annotation_text=f"Limite {limite_setor:.0%}")
            fig_comp.update_layout(
                barmode="group",
                title="Comparação da concentração setorial",
                yaxis_title="Peso do setor",
                xaxis_title="Setor",
                template="plotly_white"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Preencha os tickers na barra lateral e clique em Calcular / Atualizar dados.")

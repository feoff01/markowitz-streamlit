import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

TRADING_DAYS = 252


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
        risco = volatilidade
        sharpe = (retorno - taxa_livre_risco) / volatilidade if volatilidade > 0 else np.nan

        return {
            "pesos": pesos,
            "retorno_anual": retorno,
            "volatilidade_anual": volatilidade,
            "risco_anual": risco,
            "sharpe": sharpe
        }

    def mostrar_pesos(self, pesos):
        return pd.Series(np.array(pesos), index=self.nomes_ativos).sort_values(ascending=False)

    def otimizar_minima_variancia(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

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

    def otimizar_max_sharpe(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

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

    def otimizar_max_retorno(self, taxa_livre_risco=0.0, permitir_short=False, peso_maximo=1.0):
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

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

    def fronteira_eficiente(self, n_pontos=50, permitir_short=False, peso_maximo=1.0):
        retornos_alvo = np.linspace(self.retorno_medio_anual.min(), self.retorno_medio_anual.max(), n_pontos)
        n = len(self.nomes_ativos)
        x0 = np.array([1 / n] * n)
        bounds = None if permitir_short else tuple((0, peso_maximo) for _ in range(n))

        riscos, vols, rets = [], [], []

        for retorno_alvo in retornos_alvo:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, r=retorno_alvo: np.dot(w, self.retorno_medio_anual.values) - r}
            ]

            resultado = minimize(
                fun=lambda w: np.dot(w.T, np.dot(self.cov_anual.values, w)),
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints
            )

            if resultado.success:
                vol = np.sqrt(np.dot(resultado.x.T, np.dot(self.cov_anual.values, resultado.x)))
                vols.append(vol)
                riscos.append(vol)
                rets.append(retorno_alvo)

        return pd.DataFrame({
            "risco": riscos,
            "volatilidade": vols,
            "retorno": rets
        })

    def gerar_carteiras_aleatorias(self, numero_carteiras=3000, taxa_livre_risco=0.0):
        n = len(self.nomes_ativos)
        resultados = []

        for _ in range(numero_carteiras):
            pesos = np.random.random(n)
            pesos /= pesos.sum()

            retorno = float(np.dot(pesos, self.retorno_medio_anual.values))
            vol = float(np.sqrt(np.dot(pesos.T, np.dot(self.cov_anual.values, pesos))))
            risco = vol
            sharpe = (retorno - taxa_livre_risco) / vol if vol > 0 else np.nan

            resultados.append({
                "retorno": retorno,
                "volatilidade": vol,
                "risco": risco,
                "sharpe": sharpe
            })

        return pd.DataFrame(resultados)


def pct(x):
    return f"{x:.2%}"


st.set_page_config(page_title="Otimizador Markowitz", layout="wide")
st.title("Otimizador de Carteira de Markowitz")

if "dados_calculados" not in st.session_state:
    st.session_state.dados_calculados = False

with st.sidebar:
    st.header("Configurações")

    tickers_input = st.text_area(
        "Tickers separados por vírgula",
        value="PETR4, VALE3, ITUB4, WEGE3, BBDC4"
    )

    data_inicio = st.date_input("Data inicial", value=pd.to_datetime("2018-01-01"))
    data_fim = st.date_input("Data final", value=pd.to_datetime("today"))
    taxa_livre_risco = st.number_input("Taxa livre de risco anual", value=0.10, step=0.01)
    permitir_short = st.checkbox("Permitir short", value=False)
    peso_maximo = st.slider("Peso máximo por ativo", 0.1, 1.0, 1.0, 0.05)
    numero_carteiras = st.slider("Carteiras aleatórias", 500, 10000, 3000, 500)

    calcular = st.button("Calcular / Atualizar dados")

if calcular:
    try:
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Digite pelo menos 2 tickers.")
            st.stop()

        with st.spinner("Baixando dados e calculando..."):
            carteira = CarteiraMarkowitz(tickers, start=str(data_inicio), end=str(data_fim))
            carteira.baixar_dados()
            carteira.calcular_metricas(taxa_livre_risco=taxa_livre_risco)

            st.session_state.carteira = carteira
            st.session_state.taxa_livre_risco = taxa_livre_risco
            st.session_state.permitir_short = permitir_short
            st.session_state.peso_maximo = peso_maximo
            st.session_state.numero_carteiras = numero_carteiras
            st.session_state.dados_calculados = True

    except Exception as e:
        st.exception(e)

if st.session_state.dados_calculados:
    carteira = st.session_state.carteira
    taxa_livre_risco = st.session_state.taxa_livre_risco
    permitir_short = st.session_state.permitir_short
    peso_maximo = st.session_state.peso_maximo
    numero_carteiras = st.session_state.numero_carteiras

    if carteira.falhas_download:
        with st.expander("Tickers com falha no download"):
            for ticker, erro in carteira.falhas_download:
                st.write(f"{ticker}: {erro}")

    ativos = carteira.nomes_ativos
    n = len(ativos)

    st.subheader("Ativos válidos")
    st.write(", ".join(ativos))

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
    cols = st.columns(n)
    pesos_usuario = []

    for i, ativo in enumerate(ativos):
        chave = f"peso_{ativo}"
        if chave not in st.session_state:
            st.session_state[chave] = round(1 / n, 4)

        with cols[i]:
            p = st.number_input(
                ativo,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=chave
            )
            pesos_usuario.append(p)

    soma = sum(pesos_usuario)
    st.write(f"Soma dos pesos: {soma:.4f}")

    stats_usuario = None
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
    else:
        st.warning("A soma dos pesos deve ser 1 para calcular a sua carteira.")

    min_var = carteira.otimizar_minima_variancia(taxa_livre_risco, permitir_short, peso_maximo)
    max_sharpe = carteira.otimizar_max_sharpe(taxa_livre_risco, permitir_short, peso_maximo)
    max_retorno = carteira.otimizar_max_retorno(taxa_livre_risco, permitir_short, peso_maximo)

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
    fronteira = carteira.fronteira_eficiente(50, permitir_short, peso_maximo)
    aleatorias = carteira.gerar_carteiras_aleatorias(numero_carteiras, taxa_livre_risco)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=aleatorias["risco"],
        y=aleatorias["retorno"],
        mode="markers",
        marker=dict(size=5, color=aleatorias["sharpe"], colorscale="Viridis", showscale=True),
        name="Carteiras aleatórias"
    ))

    fig.add_trace(go.Scatter(
        x=fronteira["risco"],
        y=fronteira["retorno"],
        mode="lines",
        name="Fronteira eficiente"
    ))

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

else:
    st.info("Preencha os tickers na barra lateral e clique em Calcular / Atualizar dados.")
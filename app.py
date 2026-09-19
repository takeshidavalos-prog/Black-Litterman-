"""Aplicación Streamlit del modelo Black-Litterman.

La interfaz es deliberadamente delgada: recoge la configuración, llama a
`blmodel.pipeline` y presenta el resultado. Toda la lógica del modelo vive en
el paquete, de modo que pueda usarse, probarse y auditarse sin Streamlit.
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Localización del paquete
# ---------------------------------------------------------------------------
# Streamlit Cloud ejecuta el script desde el directorio del repositorio, pero el
# directorio del propio script no siempre queda en sys.path. Se inserta a mano
# para que `import blmodel` funcione tanto en local como en el despliegue, y
# tanto si app.py está en la raíz como en una subcarpeta.
_APP_DIR = Path(__file__).resolve().parent
for _candidate in (_APP_DIR, _APP_DIR.parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))


def _diagnose_missing_module(exc: ModuleNotFoundError) -> None:
    """Explica en pantalla qué falta, en lugar del error censurado de Streamlit.

    El error original de Streamlit Cloud llega redactado ("The original error
    message is redacted to prevent data leaks"), así que sin esto no hay forma
    de saber qué archivo falta.
    """
    st.set_page_config(page_title="Black-Litterman — error de instalación", page_icon="⚠️")
    st.title("Falta el módulo del modelo")
    st.error(f"No se pudo importar `{exc.name}`.", icon="🚫")

    presentes = sorted(p.name for p in _APP_DIR.glob("*"))
    st.markdown("### Archivos que la aplicación encontró junto a `app.py`")
    st.code("\n".join(presentes) or "(ninguno)", language="text")

    st.markdown(
        """
### Cómo se corrige

El repositorio necesita exactamente tres archivos, los tres en la **raíz**:

```
raíz-del-repositorio/
├── app.py
├── blmodel.py
└── requirements.txt
```

Si en la lista de arriba no aparece `blmodel.py`, ése es el archivo que falta.
No va dentro de ninguna carpeta: va al mismo nivel que `app.py`.

Para subirlo: en GitHub, *Add file → Upload files*, y arrastre `blmodel.py`.
        """
    )
    st.stop()


try:
    from blmodel import portfolio_return, portfolio_volatility
    from blmodel import (
        plot_allocation,
        plot_confidence_sensitivity,
        plot_correlation_heatmap,
        plot_efficient_frontier,
        plot_returns_comparison,
        plot_risk_contribution,
        plot_tau_sensitivity,
        plot_view_absorption,
        plot_view_impact,
        plot_weights_comparison,
    )
except ModuleNotFoundError as _exc:  # el módulo no llegó al despliegue
    _diagnose_missing_module(_exc)

from blmodel import (
    CovConfig,
    CovMethod,
    DataConfig,
    DeltaMethod,
    Frequency,
    GroupConstraint,
    ModelConfig,
    Objective,
    OmegaMethod,
    OptimizerConfig,
    PriorConfig,
    ReturnType,
    TauMethod,
    ViewKind,
    ViewsConfig,
    ViewSpec,
    WeightSource,
)
from blmodel import annualized_volatilities, cov_to_corr
from blmodel import DataError
from blmodel import (
    build_csv,
    build_excel,
    export_config,
    import_config,
    views_from_csv,
    views_template_csv,
    views_to_csv,
)
from blmodel import OptimizationError, unconstrained_weights
from blmodel import (
    confidence_sensitivity_analysis,
    covariance_comparison,
    load_market_data,
    prior_comparison,
    run_model,
    tau_sensitivity_analysis,
)

st.set_page_config(
    page_title="Black-Litterman",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Etiquetas legibles para los enumerados
# --------------------------------------------------------------------------
COV_LABELS = {
    CovMethod.SAMPLE: "Muestral (histórica)",
    CovMethod.LEDOIT_WOLF: "Ledoit-Wolf (shrinkage)",
    CovMethod.OAS: "Oracle Approximating Shrinkage",
    CovMethod.EWMA: "EWMA / RiskMetrics",
    CovMethod.SEMI: "Semicovarianza (riesgo a la baja)",
}
WEIGHT_LABELS = {
    WeightSource.MARKET_CAP: "Capitalización de mercado",
    WeightSource.BENCHMARK: "Pesos del benchmark",
    WeightSource.CUSTOM: "Pesos propios",
    WeightSource.EQUAL: "Equiponderado",
}
OBJECTIVE_LABELS = {
    Objective.MAX_SHARPE: "Máxima razón de Sharpe",
    Objective.MIN_VARIANCE: "Mínima varianza",
    Objective.MAX_UTILITY: "Máxima utilidad media-varianza",
    Objective.TARGET_VOL: "Volatilidad objetivo",
    Objective.TARGET_RETURN: "Rendimiento objetivo",
}
OMEGA_LABELS = {
    OmegaMethod.IDZOREK: "Idzorek (confianza 0–100%)",
    OmegaMethod.HE_LITTERMAN: "He-Litterman (proporcional)",
    OmegaMethod.MANUAL: "Ω manual",
}
FREQ_LABELS = {
    Frequency.DAILY: "Diaria",
    Frequency.WEEKLY: "Semanal",
    Frequency.MONTHLY: "Mensual",
}
RF_TICKERS = {
    "^IRX": "CETES/T-Bill 13 semanas (^IRX)",
    "^FVX": "Bono 5 años EE.UU. (^FVX)",
    "^TNX": "Bono 10 años EE.UU. (^TNX)",
    "^TYX": "Bono 30 años EE.UU. (^TYX)",
}

VIEW_COLUMNS = ["Activa", "Tipo", "Activos", "Q (%)", "Confianza (%)", "Etiqueta"]


# --------------------------------------------------------------------------
# Estado
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "views_table": pd.DataFrame(columns=VIEW_COLUMNS),
        "run": None,
        "market_data": None,
        "theme": "light",
        "last_error": None,
        "pending_config": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# --------------------------------------------------------------------------
# Caché de datos
# --------------------------------------------------------------------------
def data_cache_key(cfg: ModelConfig) -> str:
    """Sólo los campos que realmente determinan la descarga."""
    return json.dumps(
        {
            "tickers": sorted(cfg.data.tickers),
            "benchmark": cfg.data.benchmark,
            "start": cfg.data.start.isoformat() if cfg.data.start else None,
            "end": cfg.data.end.isoformat() if cfg.data.end else None,
            "lookback": cfg.data.lookback_years,
            "freq": cfg.data.frequency.value,
            "ret": cfg.data.return_type.value,
            "min_obs": cfg.data.min_observations,
            "max_missing": cfg.data.max_missing_pct,
            "rf_ticker": cfg.prior.rf_ticker,
            "rf_auto": cfg.prior.rf_auto,
            "rf_manual": cfg.prior.rf_manual,
        },
        sort_keys=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data_by_key(key: str, config_json: str, fetch_caps: bool):
    """Descarga cacheada por una hora.

    La clave es sólo lo que determina la descarga, de modo que cambiar el
    objetivo del optimizador o una view no vuelva a golpear a Yahoo Finance.
    """
    cfg = ModelConfig.from_json(config_json)
    return load_market_data(cfg, fetch_caps=fetch_caps)


PERCENT_ROWS = {
    "Rendimiento esperado", "Volatilidad", "Mayor posición", "Top 5",
}


def format_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Formatea la tabla comparativa: porcentajes donde corresponde, razones donde no."""
    out = df.copy().astype(object)
    for row in df.index:
        as_pct = row in PERCENT_ROWS
        for col in df.columns:
            value = df.loc[row, col]
            if value is None or not np.isfinite(value):
                out.loc[row, col] = "—"
            else:
                out.loc[row, col] = f"{value:.2%}" if as_pct else f"{value:,.2f}"
    return out


# --------------------------------------------------------------------------
# Formateo
# --------------------------------------------------------------------------
def pct(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}%}"


def num(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}"


def style_percent(df: pd.DataFrame, columns: list[str] | None = None, decimals: int = 2):
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df.style.format(dict.fromkeys(cols, f"{{:.{decimals}%}}"))


# --------------------------------------------------------------------------
# Barra lateral: configuración
# --------------------------------------------------------------------------
def build_config() -> ModelConfig:
    st.sidebar.title("Configuración")

    # -- Importar configuración -------------------------------------------
    with st.sidebar.expander("Cargar configuración guardada"):
        uploaded = st.file_uploader("Archivo JSON", type=["json"], key="cfg_upload")
        if uploaded is not None and st.button("Aplicar configuración", key="apply_cfg"):
            try:
                st.session_state["pending_config"] = import_config(uploaded.getvalue())
                st.success("Configuración cargada. Vuelva a ejecutar el modelo.")
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")

    preset: ModelConfig | None = st.session_state.get("pending_config")

    # -- Universo ----------------------------------------------------------
    st.sidebar.subheader("Universo")
    default_tickers = (
        ", ".join(preset.data.tickers) if preset else "SPY, EFA, EEM, AGG, TLT, GLD, VNQ"
    )
    tickers_raw = st.sidebar.text_area(
        "Tickers de Yahoo Finance",
        value=default_tickers,
        height=80,
        help="Separados por coma o salto de línea. Ejemplos: AAPL, MSFT, WALMEX.MX, IVV, BTC-USD.",
    )
    tickers = [t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]

    benchmark = st.sidebar.text_input(
        "Benchmark",
        value=preset.data.benchmark if preset else "^GSPC",
        help="Se usa para estimar δ. Ejemplos: ^GSPC (S&P 500), ^MXX (IPC), ACWI.",
    )

    # -- Datos -------------------------------------------------------------
    st.sidebar.subheader("Datos históricos")
    freq = st.sidebar.selectbox(
        "Frecuencia",
        list(Frequency),
        index=list(Frequency).index(preset.data.frequency) if preset else 0,
        format_func=lambda f: FREQ_LABELS[f],
    )
    use_dates = st.sidebar.checkbox("Definir fechas exactas", value=bool(preset and preset.data.start))
    start = end = None
    lookback = 5.0
    if use_dates:
        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input(
            "Desde", value=(preset.data.start if preset and preset.data.start else date.today() - timedelta(days=1825))
        )
        end = col2.date_input("Hasta", value=(preset.data.end if preset and preset.data.end else date.today()))
    else:
        lookback = st.sidebar.slider(
            "Ventana (años)", 1.0, 20.0,
            value=float(preset.data.lookback_years) if preset else 5.0, step=0.5,
        )

    ret_type = st.sidebar.radio(
        "Tipo de rendimiento", list(ReturnType),
        index=list(ReturnType).index(preset.data.return_type) if preset else 0,
        format_func=lambda r: "Simples" if r is ReturnType.SIMPLE else "Logarítmicos",
        horizontal=True,
    )

    with st.sidebar.expander("Reglas de validación"):
        min_obs = st.number_input("Mínimo de observaciones", 10, 2000,
                                  value=preset.data.min_observations if preset else 60, step=10)
        max_missing = st.slider("Máximo de datos faltantes", 0.0, 0.50,
                                value=float(preset.data.max_missing_pct) if preset else 0.10, step=0.01)
        outlier_sigma = st.slider("Umbral de atípicos (σ)", 3.0, 15.0,
                                  value=float(preset.data.outlier_sigma) if preset else 8.0, step=0.5)

    # -- Covarianza --------------------------------------------------------
    st.sidebar.subheader("Matriz de covarianzas")
    cov_method = st.sidebar.selectbox(
        "Estimador", list(CovMethod),
        index=list(CovMethod).index(preset.covariance.method) if preset else 1,
        format_func=lambda m: COV_LABELS[m],
    )
    ewma_lambda = 0.94
    if cov_method is CovMethod.EWMA:
        ewma_lambda = st.sidebar.slider(
            "λ (decaimiento)", 0.80, 0.99,
            value=float(preset.covariance.ewma_lambda) if preset else 0.94, step=0.01,
            help="Menor λ reacciona más rápido a cambios de régimen, con más ruido.",
        )

    # -- Prior -------------------------------------------------------------
    st.sidebar.subheader("Prior de equilibrio")
    weight_source = st.sidebar.selectbox(
        "Pesos de mercado (w)", list(WeightSource),
        index=list(WeightSource).index(preset.prior.weight_source) if preset else 0,
        format_func=lambda w: WEIGHT_LABELS[w],
    )
    custom_weights: dict[str, float] = {}
    if weight_source is WeightSource.CUSTOM:
        raw = st.sidebar.text_area(
            "Pesos propios",
            value="\n".join(f"{t}: {1/len(tickers):.4f}" for t in tickers) if tickers else "",
            height=100, help="Un renglón por activo, formato TICKER: peso. Se normalizan a 1.",
        )
        for line in raw.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                with contextlib.suppress(ValueError):
                    custom_weights[k.strip().upper()] = float(v)

    delta_method = st.sidebar.radio(
        "Aversión al riesgo (δ)", list(DeltaMethod),
        index=list(DeltaMethod).index(preset.prior.delta_method) if preset else 0,
        format_func=lambda d: "Implícita del benchmark" if d is DeltaMethod.IMPLIED else "Manual",
        horizontal=True,
    )
    delta_manual = st.sidebar.slider(
        "δ manual", 0.5, 10.0,
        value=float(preset.prior.delta_manual) if preset else 2.5, step=0.1,
        disabled=delta_method is DeltaMethod.IMPLIED,
        help="Valor convencional: 2.5. Mayor δ ⇒ portafolio más conservador.",
    )

    tau_method = st.sidebar.radio(
        "Incertidumbre del prior (τ)", list(TauMethod),
        index=list(TauMethod).index(preset.prior.tau_method) if preset else 0,
        format_func=lambda t: "Manual" if t is TauMethod.MANUAL else "τ = 1/T (Meucci)",
        horizontal=True,
    )
    tau_manual = st.sidebar.slider(
        "τ manual", 0.005, 1.0,
        value=float(preset.prior.tau_manual) if preset else 0.05, step=0.005, format="%.3f",
        disabled=tau_method is TauMethod.ONE_OVER_T,
    )

    rf_auto = st.sidebar.checkbox(
        "Tasa libre de riesgo automática",
        value=preset.prior.rf_auto if preset else True,
    )
    rf_ticker, rf_manual = "^IRX", None
    if rf_auto:
        rf_ticker = st.sidebar.selectbox(
            "Instrumento de referencia", list(RF_TICKERS),
            format_func=lambda t: RF_TICKERS[t],
            index=list(RF_TICKERS).index(preset.prior.rf_ticker) if preset and preset.prior.rf_ticker in RF_TICKERS else 0,
        )
    else:
        rf_manual = st.sidebar.number_input(
            "Tasa libre de riesgo anual (%)", -2.0, 30.0,
            value=float((preset.prior.rf_manual or 0.04) * 100) if preset else 4.0, step=0.25,
        ) / 100.0

    # -- Optimizador -------------------------------------------------------
    st.sidebar.subheader("Optimización")
    objective = st.sidebar.selectbox(
        "Objetivo", list(Objective),
        index=list(Objective).index(preset.optimizer.objective) if preset else 0,
        format_func=lambda o: OBJECTIVE_LABELS[o],
    )
    target_vol, target_return = 0.15, 0.10
    if objective is Objective.TARGET_VOL:
        target_vol = st.sidebar.slider("Volatilidad objetivo", 0.02, 0.60,
                                       value=float(preset.optimizer.target_vol) if preset else 0.15, step=0.01)
    if objective is Objective.TARGET_RETURN:
        target_return = st.sidebar.slider("Rendimiento objetivo", -0.05, 0.50,
                                          value=float(preset.optimizer.target_return) if preset else 0.10, step=0.005)

    allow_short = st.sidebar.checkbox("Permitir posiciones cortas",
                                      value=preset.optimizer.allow_short if preset else False)
    col1, col2 = st.sidebar.columns(2)
    weight_lower = col1.number_input("Peso mínimo", -2.0, 1.0,
                                     value=float(preset.optimizer.weight_lower) if preset else (-0.30 if allow_short else 0.0),
                                     step=0.05, disabled=not allow_short)
    weight_upper = col2.number_input("Peso máximo", 0.01, 2.0,
                                     value=float(preset.optimizer.weight_upper) if preset else 1.0, step=0.05)
    max_leverage = 1.0
    if allow_short:
        max_leverage = st.sidebar.slider("Apalancamiento bruto máximo (‖w‖₁)", 1.0, 3.0,
                                         value=float(preset.optimizer.max_gross_leverage) if preset else 1.5, step=0.1)

    asset_bounds: dict[str, tuple[float, float]] = {}
    with st.sidebar.expander("Cotas por activo"):
        st.caption("Formato: TICKER: mínimo, máximo — un renglón por activo.")
        default_bounds = "\n".join(
            f"{k}: {v[0]}, {v[1]}" for k, v in (preset.optimizer.asset_bounds if preset else {}).items()
        )
        raw_bounds = st.text_area("Cotas", value=default_bounds, height=80, label_visibility="collapsed")
        for line in raw_bounds.splitlines():
            if ":" in line:
                ticker, _, rest = line.partition(":")
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) == 2:
                    with contextlib.suppress(ValueError):
                        asset_bounds[ticker.strip().upper()] = (float(parts[0]), float(parts[1]))

    groups: list[GroupConstraint] = []
    with st.sidebar.expander("Restricciones por grupo"):
        st.caption("Formato: Nombre | TICKER1, TICKER2 | mínimo | máximo")
        default_groups = "\n".join(
            f"{g.name} | {', '.join(g.members)} | {g.lower} | {g.upper}"
            for g in (preset.optimizer.groups if preset else [])
        )
        raw_groups = st.text_area("Grupos", value=default_groups, height=80, label_visibility="collapsed")
        for line in raw_groups.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 4:
                with contextlib.suppress(ValueError):
                    groups.append(GroupConstraint(
                        name=parts[0],
                        members=[m.strip().upper() for m in parts[1].split(",") if m.strip()],
                        lower=float(parts[2]), upper=float(parts[3]),
                    ))

    frontier_points = st.sidebar.slider("Puntos de la frontera", 10, 80,
                                        value=preset.optimizer.frontier_points if preset else 40, step=5)

    # -- Views -------------------------------------------------------------
    st.sidebar.subheader("Views")
    omega_method = st.sidebar.selectbox(
        "Método de Ω", list(OmegaMethod),
        index=list(OmegaMethod).index(preset.views.method) if preset else 0,
        format_func=lambda m: OMEGA_LABELS[m],
        help=(
            "Idzorek traduce una confianza de 0–100% a Ω, y es el estándar de "
            "industria. He-Litterman fija Ω por la estructura de Σ, sin parámetro "
            "subjetivo: todas las views pesan igual por construcción."
        ),
    )
    with st.sidebar.expander("Cargar views desde CSV"):
        up = st.file_uploader("Archivo CSV", type=["csv"], key="views_upload")
        if up is not None and st.button("Importar views", key="import_views"):
            imported, errors = views_from_csv(up.getvalue())
            if imported:
                st.session_state["views_table"] = views_to_table(imported)
                st.success(f"{len(imported)} views importadas.")
            for e in errors:
                st.error(e)
        st.download_button("Descargar plantilla", views_template_csv(),
                           file_name="plantilla_views.csv", mime="text/csv",
                           width="stretch")

    # -- Apariencia --------------------------------------------------------
    st.sidebar.subheader("Apariencia")
    theme = st.sidebar.radio("Tema de las gráficas", ["light", "dark"],
                             format_func=lambda t: "Claro" if t == "light" else "Oscuro",
                             horizontal=True, index=0 if st.session_state["theme"] == "light" else 1)
    st.session_state["theme"] = theme

    return ModelConfig(
        data=DataConfig(
            tickers=tickers, benchmark=benchmark.strip(), start=start, end=end,
            lookback_years=lookback, frequency=freq, return_type=ret_type,
            min_observations=int(min_obs), max_missing_pct=float(max_missing),
            outlier_sigma=float(outlier_sigma),
        ),
        covariance=CovConfig(method=cov_method, ewma_lambda=float(ewma_lambda)),
        prior=PriorConfig(
            weight_source=weight_source, custom_weights=custom_weights,
            delta_method=delta_method, delta_manual=float(delta_manual),
            tau_method=tau_method, tau_manual=float(tau_manual),
            rf_ticker=rf_ticker, rf_manual=rf_manual, rf_auto=rf_auto,
        ),
        views=ViewsConfig(method=omega_method, views=views_from_table(st.session_state["views_table"])),
        optimizer=OptimizerConfig(
            objective=objective, allow_short=allow_short,
            weight_lower=float(weight_lower) if allow_short else 0.0,
            weight_upper=float(weight_upper), asset_bounds=asset_bounds, groups=groups,
            max_gross_leverage=float(max_leverage), target_vol=float(target_vol),
            target_return=float(target_return), frontier_points=int(frontier_points),
        ),
    )


# --------------------------------------------------------------------------
# Views: tabla ↔ objetos
# --------------------------------------------------------------------------
def views_from_table(table: pd.DataFrame) -> list[ViewSpec]:
    out: list[ViewSpec] = []
    if table is None or table.empty:
        return out
    for _, row in table.iterrows():
        assets: dict[str, float] = {}
        for part in str(row.get("Activos", "")).split(";"):
            part = part.strip()
            if not part:
                continue
            ticker, _, weight = part.partition(":")
            try:
                assets[ticker.strip().upper()] = float(weight) if weight.strip() else 1.0
            except ValueError:
                continue
        if not assets:
            continue
        kind = ViewKind.RELATIVE if str(row.get("Tipo", "")).lower().startswith("rel") else ViewKind.ABSOLUTE
        try:
            q = float(row.get("Q (%)", 0.0)) / 100.0
            conf = float(row.get("Confianza (%)", 50.0)) / 100.0
        except (TypeError, ValueError):
            continue
        out.append(ViewSpec(
            kind=kind, assets=assets, q=q, confidence=conf,
            label=str(row.get("Etiqueta", "") or ""),
            enabled=bool(row.get("Activa", True)),
        ))
    return out


def views_to_table(views: list[ViewSpec]) -> pd.DataFrame:
    rows = [
        {
            "Activa": v.enabled,
            "Tipo": "Relativa" if v.kind is ViewKind.RELATIVE else "Absoluta",
            "Activos": ";".join(f"{a}:{w:g}" for a, w in v.assets.items()),
            "Q (%)": v.q * 100,
            "Confianza (%)": v.confidence * 100,
            "Etiqueta": v.label,
        }
        for v in views
    ]
    return pd.DataFrame(rows, columns=VIEW_COLUMNS)


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------
def section_summary(run) -> None:
    p = run.portfolio
    theme = st.session_state["theme"]

    st.subheader("Resultado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendimiento esperado", pct(p.expected_return))
    c2.metric("Volatilidad", pct(p.volatility))
    c3.metric("Razón de Sharpe", num(p.sharpe))
    c4.metric("Activos con posición", f"{int((p.weights.abs() > 1e-6).sum())} de {len(p.weights)}")

    st.caption(
        f"Objetivo: {p.objective} · Ω: {run.bl.method} · Σ: {COV_LABELS[run.config.covariance.method]} · "
        f"δ = {run.prior.delta:.2f} · τ = {run.prior.tau:.4f} · r_f = {pct(run.prior.risk_free)}"
    )

    if run.notes:
        with st.expander(f"Advertencias y supuestos ({len(run.notes)})", expanded=len(run.notes) <= 3):
            for note in run.notes:
                st.warning(note, icon="⚠️")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(plot_allocation(p.weights, theme), width="stretch")
    with right:
        st.markdown("**Pesos**")
        table = pd.DataFrame({
            "Peso": p.weights,
            "Peso de mercado": run.prior.weights,
            "Diferencia": p.weights - run.prior.weights,
        }).sort_values("Peso", ascending=False)
        st.dataframe(style_percent(table), width="stretch", height=380)

    if run.comparison is not None:
        st.markdown("**Comparativa de portafolios, con los mismos insumos**")
        st.dataframe(format_comparison(run.comparison), width="stretch")
        st.caption(
            "Todos los portafolios se evalúan con el mismo μ_BL y la misma Σ, "
            "así que la comparación aísla el efecto de la asignación."
        )


def section_data(run) -> None:
    rep = run.validation
    st.subheader("Datos y validación")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos aceptados", f"{len(rep.accepted)} de {len(rep.requested)}")
    c2.metric("Observaciones", f"{rep.n_observations:,}")
    c3.metric("Desde", rep.start.isoformat() if rep.start else "—")
    c4.metric("Hasta", rep.end.isoformat() if rep.end else "—")

    st.dataframe(rep.to_frame(), width="stretch", hide_index=True)

    for w in rep.warnings:
        st.warning(w, icon="⚠️")

    st.markdown("**Diagnóstico de Σ**")
    d = run.cov_diag
    c1, c2, c3 = st.columns(3)
    c1.metric("Número de condición", f"{d.condition_number:,.0f}")
    c2.metric("Menor valor propio", f"{d.min_eigenvalue:.2e}")
    c3.metric("Intensidad de shrinkage", num(d.shrinkage) if d.shrinkage is not None else "No aplica")
    for note in d.notes():
        st.info(note, icon="ℹ️")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            plot_correlation_heatmap(cov_to_corr(run.cov), st.session_state["theme"]),
            width="stretch",
        )
    with right:
        st.markdown("**Volatilidad anualizada por estimador**")
        comp = covariance_comparison(run)
        comp.columns = [COV_LABELS.get(CovMethod(c), c) for c in comp.columns]
        st.dataframe(style_percent(comp), width="stretch")
        st.caption(
            "Contraste entre estimadores. Diferencias grandes entre columnas "
            "indican que la elección del estimador sí mueve el resultado."
        )

    with st.expander("Serie de precios"):
        st.dataframe(run.prices.tail(250), width="stretch")


def section_prior(run, market_data) -> None:
    st.subheader("Prior de equilibrio")
    st.markdown(
        "El prior es el rendimiento que **ya está implícito en los precios**: "
        "los rendimientos esperados que justificarían el portafolio de mercado actual. "
        "Se obtiene invirtiendo la optimización de Markowitz: **Π = δ · Σ · w**."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("δ (aversión al riesgo)", num(run.prior.delta), help=run.prior.delta_source)
    c2.metric("τ (incertidumbre del prior)", f"{run.prior.tau:.4f}", help=run.prior.tau_source)
    c3.metric("Tasa libre de riesgo", pct(run.prior.risk_free), help=run.prior.rf_source)
    st.caption(f"{run.prior.delta_source} · {run.prior.tau_source} · {run.prior.rf_source}")

    st.info(f"Pesos de mercado: **{run.prior.weight_source}**.", icon="ℹ️")
    for fb in run.prior.fallbacks:
        st.warning(fb, icon="⚠️")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Π resultante según la fuente de pesos**")
        comp = prior_comparison(run, market_data)
        if not comp.empty:
            st.dataframe(style_percent(comp), width="stretch")
            st.caption(
                "Mismo δ y misma Σ; sólo cambia w. Es la respuesta a "
                "«¿y si los pesos de referencia fueran otros?»."
            )
    with right:
        st.markdown("**Prior vigente**")
        table = pd.DataFrame({
            "Peso de mercado": run.prior.weights,
            "Π": run.prior.pi,
            "Volatilidad": annualized_volatilities(run.cov),
        })
        st.dataframe(style_percent(table), width="stretch")

    if run.prior.market_caps is not None and len(run.prior.market_caps):
        with st.expander("Capitalización de mercado descargada"):
            caps = run.prior.market_caps.reindex(run.assets).to_frame("Capitalización (USD)")
            st.dataframe(caps.style.format("{:,.0f}"), width="stretch")


def section_views(run) -> None:
    st.subheader("Views del inversionista")

    if run.views.n_views == 0:
        st.info(
            "Sin views activas, el posterior es idéntico al prior y el modelo "
            "devuelve el portafolio de equilibrio. Agregue views en el panel de la "
            "izquierda para desviarlo.",
            icon="ℹ️",
        )
        for w in run.views.warnings:
            st.warning(w, icon="⚠️")
        return

    st.caption(f"Método de Ω: **{run.views.method}**")

    diag = run.view_diag
    if diag is not None:
        display = diag.copy()
        st.dataframe(
            display.style.format({
                "Q (su view)": "{:.2%}",
                "P·Π (equilibrio)": "{:.2%}",
                "P·μ_BL (posterior)": "{:.2%}",
                "Desacuerdo (Q − P·Π)": "{:+.2%}",
                "Absorción": "{:.1%}",
                "Confianza declarada": "{:.0%}",
                "Ω": "{:.2e}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "**Absorción**: qué proporción del desacuerdo con el equilibrio incorporó "
            "el modelo. Con confianza 100% tiende a 100%; con 0%, a 0%."
        )
        st.plotly_chart(plot_view_absorption(diag, st.session_state["theme"]),
                        width="stretch")

    with st.expander("Matrices P y Q (auditoría)"):
        st.markdown("**P — selección de activos por view**")
        st.dataframe(run.views.p_frame().style.format("{:+.4f}"), width="stretch")
        st.markdown("**Q y Ω**")
        st.dataframe(
            run.views.summary_frame().style.format({
                "Q (rendimiento)": "{:.2%}", "Confianza": "{:.0%}",
                "Ω (varianza del error)": "{:.3e}",
            }),
            width="stretch", hide_index=True,
        )

    for w in run.views.warnings:
        st.warning(w, icon="⚠️")


def section_posterior(run) -> None:
    theme = st.session_state["theme"]
    st.subheader("Posterior: equilibrio combinado con sus views")
    st.markdown(
        "μ_BL = Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ) — el promedio del equilibrio y de "
        "sus opiniones, ponderado por la precisión de cada fuente."
    )

    st.plotly_chart(plot_returns_comparison(run.prior.pi, run.bl.mu_bl, theme),
                    width="stretch")

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Tabla comparativa**")
        st.dataframe(style_percent(run.bl.comparison_frame()), width="stretch", height=380)
    with right:
        st.plotly_chart(plot_view_impact(run.bl.view_impact, theme), width="stretch")

    if run.bl.n_views > 0:
        st.caption(
            f"Desvío agregado sobre el equilibrio: {run.bl.total_view_impact:.2%} "
            "(suma de los valores absolutos de μ_BL − Π)."
        )


def section_optimization(run) -> None:
    theme = st.session_state["theme"]
    st.subheader("Optimización")

    for d in run.portfolio.diagnostics:
        st.info(d, icon="ℹ️")

    weights = {
        "Mercado": run.prior.weights,
        "Black-Litterman": run.portfolio.weights,
    }
    st.plotly_chart(plot_weights_comparison(weights, theme), width="stretch")

    left, right = st.columns(2)
    with left:
        if run.frontier is not None and not run.frontier.empty:
            marks = {
                "Black-Litterman": (run.portfolio.volatility, run.portfolio.expected_return),
                "Mercado": (
                    portfolio_volatility(run.prior.weights, run.bl.cov_bl),
                    portfolio_return(run.prior.weights, run.bl.mu_bl),
                ),
            }
            st.plotly_chart(plot_efficient_frontier(run.frontier, marks, theme),
                            width="stretch")
    with right:
        if run.risk_contrib is not None:
            st.plotly_chart(plot_risk_contribution(run.risk_contrib, theme),
                            width="stretch")

    if run.risk_contrib is not None:
        st.markdown("**Descomposición del riesgo**")
        rc = run.risk_contrib[run.risk_contrib["Peso"].abs() > 1e-6].sort_values(
            "% del riesgo", ascending=False
        )
        st.dataframe(style_percent(rc, ["Peso", "Contribución al riesgo", "% del riesgo"]),
                     width="stretch")
        st.caption(
            "Dónde está el capital y de dónde viene el riesgo rara vez coinciden. "
            "Las contribuciones suman exactamente la volatilidad del portafolio."
        )

    with st.expander("Referencia sin restricciones: w = (δΣ)⁻¹μ"):
        unc = unconstrained_weights(run.bl.mu_bl, run.bl.cov_bl, run.prior.delta)
        table = pd.DataFrame({
            "Sin restricciones": unc,
            "Con restricciones": run.portfolio.weights,
            "Costo de las restricciones": run.portfolio.weights - unc,
        })
        st.dataframe(style_percent(table), width="stretch")
        st.caption(
            "La solución cerrada del paper original. Suele traer cortos grandes "
            "y apalancamiento: sirve para medir cuánto cuesta imponer las restricciones."
        )


def section_sensitivity(run) -> None:
    theme = st.session_state["theme"]
    st.subheader("Análisis de sensibilidad")
    st.markdown(
        "τ es el parámetro más discutido del modelo y el menos observable. "
        "Si los pesos apenas se mueven en el rango razonable, la elección deja de "
        "importar; si se mueven mucho, hay que justificarla."
    )

    col1, col2, col3 = st.columns(3)
    tau_min = col1.number_input("τ mínimo", 0.001, 0.5, 0.005, step=0.005, format="%.3f")
    tau_max = col2.number_input("τ máximo", 0.01, 2.0, 1.0, step=0.05)
    n_points = col3.slider("Puntos", 6, 40, 18)

    constrained = st.checkbox(
        "Aplicar las restricciones del optimizador", value=True,
        help="Sin marcar, usa la solución cerrada (δΣ)⁻¹μ, que aísla el efecto de τ.",
    )

    if st.button("Calcular sensibilidad a τ", type="primary"):
        with st.spinner("Recorriendo τ…"):
            try:
                sens = tau_sensitivity_analysis(
                    run, float(tau_min), float(tau_max), int(n_points), constrained
                )
                st.session_state["tau_sens"] = sens
            except Exception as exc:
                st.error(f"No se pudo completar el análisis: {exc}")

    sens = st.session_state.get("tau_sens")
    if sens is not None and not sens.empty:
        st.plotly_chart(plot_tau_sensitivity(sens, theme), width="stretch")
        spread = (sens.max() - sens.min()).sort_values(ascending=False)
        st.markdown("**Rango de variación del peso en el barrido de τ**")
        st.dataframe(spread.to_frame("Amplitud").style.format("{:.2%}"), width="stretch")
        st.caption(
            f"Activo más sensible: **{spread.index[0]}**, con {spread.iloc[0]:.2%} "
            "de amplitud entre el τ mínimo y el máximo."
        )

    st.divider()

    if run.config.views.method is OmegaMethod.IDZOREK and run.views.n_views > 0:
        st.markdown("**Sensibilidad a la confianza declarada**")
        st.caption("Mueve en bloque la confianza de todas las views, de 0% a 100%.")
        if st.button("Calcular sensibilidad a la confianza"):
            with st.spinner("Recorriendo niveles de confianza…"):
                try:
                    st.session_state["conf_sens"] = confidence_sensitivity_analysis(run)
                except Exception as exc:
                    st.error(f"No se pudo completar el análisis: {exc}")
        csens = st.session_state.get("conf_sens")
        if csens is not None and not csens.empty:
            st.plotly_chart(plot_confidence_sensitivity(csens, theme), width="stretch")
            st.dataframe(style_percent(csens), width="stretch")
    else:
        st.info(
            "La sensibilidad a la confianza requiere Ω por Idzorek y al menos una "
            "view activa: es el único método donde la confianza es un parámetro explícito.",
            icon="ℹ️",
        )


def section_export(run) -> None:
    st.subheader("Exportación")
    st.caption(
        "El JSON de configuración reconstruye la corrida completa: mismos tickers, "
        "misma ventana, mismos parámetros y mismas views."
    )

    sheets = {
        "Pesos": pd.DataFrame({
            "Peso BL": run.portfolio.weights,
            "Peso de mercado": run.prior.weights,
            "Diferencia": run.portfolio.weights - run.prior.weights,
        }),
        "Rendimientos": run.bl.comparison_frame(),
        "Covarianza": run.bl.cov_bl,
        "Correlaciones": cov_to_corr(run.cov),
        "Riesgo": run.risk_contrib if run.risk_contrib is not None else pd.DataFrame(),
        "Comparativa": run.comparison if run.comparison is not None else pd.DataFrame(),
        "Validacion": run.validation.to_frame(),
    }
    if run.view_diag is not None:
        sheets["Views"] = run.view_diag
    if run.frontier is not None:
        sheets["Frontera"] = run.frontier

    percent_cols = {
        "Pesos": ["Peso BL", "Peso de mercado", "Diferencia"],
        "Rendimientos": ["Π (equilibrio)", "μ_BL (posterior)", "Δ por views"],
        "Riesgo": ["Peso", "Contribución al riesgo", "% del riesgo"],
        "Frontera": ["Rendimiento", "Volatilidad"],
    }

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Descargar Excel", build_excel(sheets, percent_cols),
        file_name=f"black_litterman_{date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    c2.download_button(
        "Descargar pesos (CSV)", build_csv(sheets["Pesos"]),
        file_name=f"pesos_bl_{date.today().isoformat()}.csv",
        mime="text/csv", width="stretch",
    )
    c3.download_button(
        "Descargar configuración (JSON)",
        export_config(run.config, extra={
            "activos": run.assets,
            "observaciones": run.validation.n_observations,
            "delta": run.prior.delta,
            "tau": run.prior.tau,
            "tasa_libre_riesgo": run.prior.risk_free,
        }),
        file_name=f"config_bl_{date.today().isoformat()}.json",
        mime="application/json", width="stretch",
    )

    st.divider()
    st.markdown("**Contenido del libro de Excel**")
    st.dataframe(
        pd.DataFrame({
            "Hoja": list(sheets),
            "Filas": [len(v) for v in sheets.values()],
            "Columnas": [len(v.columns) for v in sheets.values()],
        }),
        width="stretch", hide_index=True,
    )


# --------------------------------------------------------------------------
# Aplicación
# --------------------------------------------------------------------------
def main() -> None:
    st.title("Modelo Black-Litterman")
    st.caption(
        "Optimización de portafolios combinando el equilibrio de mercado con views "
        "propias. Datos: Yahoo Finance."
    )

    cfg = build_config()

    st.markdown("### Views")
    st.caption(
        "**Absoluta**: un activo rendirá Q. Formato de activos: `NVDA:1`. — "
        "**Relativa**: unos activos superan a otros por Q. Formato: `NVDA:1;KO:-1` "
        "(los pesos deben sumar cero)."
    )

    edited = st.data_editor(
        st.session_state["views_table"],
        num_rows="dynamic",
        width="stretch",
        key="views_editor",
        column_config={
            "Activa": st.column_config.CheckboxColumn("Activa", default=True, width="small"),
            "Tipo": st.column_config.SelectboxColumn("Tipo", options=["Absoluta", "Relativa"],
                                                     default="Absoluta", width="small"),
            "Activos": st.column_config.TextColumn(
                "Activos", help="TICKER:peso separados por punto y coma. Ej: NVDA:1;KO:-1",
                width="medium"),
            "Q (%)": st.column_config.NumberColumn("Q (%)", min_value=-100.0, max_value=200.0,
                                                   step=0.5, format="%.2f", default=10.0),
            "Confianza (%)": st.column_config.NumberColumn(
                "Confianza (%)", min_value=0.0, max_value=100.0, step=5.0,
                format="%.0f", default=50.0,
                help="Sólo aplica con Ω por Idzorek."),
            "Etiqueta": st.column_config.TextColumn("Etiqueta", width="medium"),
        },
    )
    st.session_state["views_table"] = edited

    c1, c2, _ = st.columns([1, 1, 3])
    run_clicked = c1.button("Ejecutar modelo", type="primary", width="stretch")
    if not edited.empty:
        c2.download_button(
            "Exportar views (CSV)", views_to_csv(views_from_table(edited)),
            file_name="views.csv", mime="text/csv", width="stretch",
        )

    # -- Ejecución ---------------------------------------------------------
    if run_clicked:
        st.session_state["last_error"] = None
        # Se toman las views tal como quedaron en el editor de esta misma
        # ejecución, no las del estado previo.
        cfg.views.views = views_from_table(edited)
        if len(cfg.data.tickers) < 2:
            st.session_state["last_error"] = "Especifique al menos dos tickers."
        else:
            try:
                with st.spinner("Descargando datos de Yahoo Finance…"):
                    market = cached_market_data_by_key(
                        data_cache_key(cfg), cfg.to_json(),
                        cfg.prior.weight_source is WeightSource.MARKET_CAP,
                    )
                with st.spinner("Resolviendo el modelo…"):
                    st.session_state["run"] = run_model(cfg, market)
                    st.session_state["market_data"] = market
                    st.session_state.pop("tau_sens", None)
                    st.session_state.pop("conf_sens", None)
            except DataError as exc:
                st.session_state["last_error"] = f"Problema con los datos: {exc}"
            except OptimizationError as exc:
                st.session_state["last_error"] = f"Problema de optimización: {exc}"
            except Exception as exc:  # red caída, ticker exótico, etc.
                st.session_state["last_error"] = f"Error inesperado: {exc}"

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"], icon="🚫")

    run = st.session_state.get("run")
    if run is None:
        st.info(
            "Configure el universo en el panel izquierdo, agregue sus views si las "
            "tiene, y presione **Ejecutar modelo**. Sin views, el resultado es el "
            "portafolio de equilibrio de mercado.",
            icon="👈",
        )
        with st.expander("Cómo funciona el modelo", expanded=True):
            st.markdown(
                """
**1. Prior de equilibrio.** El modelo deduce de los precios qué rendimientos
espera el mercado: **Π = δ · Σ · w**, invirtiendo la optimización de Markowitz.
Sin opiniones propias, ése es el resultado.

**2. Sus views.** Usted declara rendimientos absolutos (*"NVDA rendirá 15%"*) o
relativos (*"NVDA superará a KO por 5%"*), cada uno con una confianza.

**3. Combinación bayesiana.** El posterior μ_BL promedia ambas fuentes según su
precisión. Una view con 50% de confianza mueve el peso exactamente la mitad de
la distancia hacia donde lo llevaría con certeza total.

**4. Optimización.** El portafolio resultante se obtiene bajo las restricciones
que usted imponga: long-only, cotas por activo, límites por grupo,
apalancamiento.

La virtud del método sobre Markowitz puro: no hereda el ruido de las medias
históricas, así que no produce portafolios esquinados en el activo que
casualmente subió más.
                """
            )
        return

    tabs = st.tabs([
        "Resumen", "Datos", "Prior", "Views", "Posterior",
        "Optimización", "Sensibilidad", "Exportar",
    ])
    with tabs[0]:
        section_summary(run)
    with tabs[1]:
        section_data(run)
    with tabs[2]:
        section_prior(run, st.session_state["market_data"])
    with tabs[3]:
        section_views(run)
    with tabs[4]:
        section_posterior(run)
    with tabs[5]:
        section_optimization(run)
    with tabs[6]:
        section_sensitivity(run)
    with tabs[7]:
        section_export(run)


if __name__ == "__main__":
    main()

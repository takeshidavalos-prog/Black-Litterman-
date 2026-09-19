"""Modelo Black-Litterman — módulo único.

Contiene el modelo completo: descarga y validación de datos de Yahoo Finance,
estimadores de covarianza, prior de equilibrio, construcción de views, motor
bayesiano, optimizador convexo, métricas, gráficas y exportación.

Es la consolidación de lo que en la versión modular son doce archivos dentro
del paquete `blmodel/`. La lógica es idéntica; sólo cambia cómo está repartida
en disco, para que el despliegue requiera tres archivos en lugar de catorce.

Las secciones van en orden de dependencia y están marcadas con encabezados que
indican de qué archivo provienen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import date
from datetime import date, timedelta
from datetime import datetime
from enum import Enum
from scipy.optimize import minimize_scalar
from sklearn.covariance import OAS, LedoitWolf
from typing import Any
import cvxpy as cp
import io
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings


# ==========================================================================
# Configuración serializable de una corrida
# (originalmente blmodel/config.py)
# ==========================================================================

# --------------------------------------------------------------------------
# Enumeraciones
# --------------------------------------------------------------------------
class Frequency(str, Enum):
    """Frecuencia de muestreo de los rendimientos."""

    DAILY = "diaria"
    WEEKLY = "semanal"
    MONTHLY = "mensual"

    @property
    def yf_interval(self) -> str:
        return {"diaria": "1d", "semanal": "1wk", "mensual": "1mo"}[self.value]

    @property
    def periods_per_year(self) -> int:
        """Períodos por año usados para anualizar media y covarianza."""
        return {"diaria": 252, "semanal": 52, "mensual": 12}[self.value]


class ReturnType(str, Enum):
    SIMPLE = "simples"
    LOG = "logaritmicos"


class CovMethod(str, Enum):
    SAMPLE = "muestral"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS = "oas"
    EWMA = "ewma"
    SEMI = "semicovarianza"


class WeightSource(str, Enum):
    """Origen de los pesos de mercado usados en la optimización inversa."""

    MARKET_CAP = "capitalizacion"
    BENCHMARK = "benchmark"
    CUSTOM = "propios"
    EQUAL = "equiponderado"


class DeltaMethod(str, Enum):
    IMPLIED = "implicito"  # delta = (E[Rb] - rf) / var(Rb)
    MANUAL = "manual"


class TauMethod(str, Enum):
    MANUAL = "manual"
    ONE_OVER_T = "1_sobre_T"  # Meucci


class OmegaMethod(str, Enum):
    IDZOREK = "idzorek"
    HE_LITTERMAN = "he_litterman"
    MANUAL = "manual"


class Objective(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_varianza"
    MAX_UTILITY = "max_utilidad"
    TARGET_VOL = "vol_objetivo"
    TARGET_RETURN = "rendimiento_objetivo"


class ViewKind(str, Enum):
    ABSOLUTE = "absoluta"
    RELATIVE = "relativa"


# --------------------------------------------------------------------------
# Configuraciones
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Universo, ventana y tratamiento de los precios."""

    tickers: list[str] = field(default_factory=list)
    benchmark: str = "^GSPC"
    start: date | None = None
    end: date | None = None
    lookback_years: float = 5.0
    frequency: Frequency = Frequency.DAILY
    return_type: ReturnType = ReturnType.SIMPLE
    # Validación
    min_observations: int = 60
    max_missing_pct: float = 0.10
    outlier_sigma: float = 8.0
    drop_incomplete: bool = True


@dataclass
class CovConfig:
    method: CovMethod = CovMethod.LEDOIT_WOLF
    ewma_lambda: float = 0.94
    semi_threshold: float = 0.0
    force_psd: bool = True


@dataclass
class PriorConfig:
    weight_source: WeightSource = WeightSource.MARKET_CAP
    custom_weights: dict[str, float] = field(default_factory=dict)
    delta_method: DeltaMethod = DeltaMethod.IMPLIED
    delta_manual: float = 2.5
    tau_method: TauMethod = TauMethod.MANUAL
    tau_manual: float = 0.05
    rf_ticker: str = "^IRX"
    rf_manual: float | None = None
    rf_auto: bool = True


@dataclass
class ViewSpec:
    """Una view del inversionista.

    - Absoluta:  "<activo> rendirá q anual"       -> assets = {ticker: 1.0}
    - Relativa:  "A superará a B por q"           -> assets = {A: +1.0, B: -1.0}

    Los pesos de una view relativa deben sumar cero; los de una absoluta, uno.
    """

    kind: ViewKind = ViewKind.ABSOLUTE
    assets: dict[str, float] = field(default_factory=dict)
    q: float = 0.0
    confidence: float = 0.50  # Idzorek, en [0, 1]
    omega_manual: float | None = None
    label: str = ""
    enabled: bool = True


@dataclass
class ViewsConfig:
    method: OmegaMethod = OmegaMethod.IDZOREK
    views: list[ViewSpec] = field(default_factory=list)


@dataclass
class GroupConstraint:
    name: str
    members: list[str]
    lower: float = 0.0
    upper: float = 1.0


@dataclass
class OptimizerConfig:
    objective: Objective = Objective.MAX_SHARPE
    allow_short: bool = False
    weight_lower: float = 0.0
    weight_upper: float = 1.0
    asset_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    groups: list[GroupConstraint] = field(default_factory=list)
    max_gross_leverage: float = 1.0  # ||w||_1
    target_vol: float = 0.15
    target_return: float = 0.10
    frontier_points: int = 40


@dataclass
class ModelConfig:
    """Configuración completa de una corrida."""

    data: DataConfig = field(default_factory=DataConfig)
    covariance: CovConfig = field(default_factory=CovConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    version: str = "1.0.0"

    # -- serialización ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def _clean(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, date):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        return _clean(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        def _date(v: Any) -> date | None:
            return date.fromisoformat(v) if isinstance(v, str) else None

        data_d = dict(d.get("data", {}))
        data_d["frequency"] = Frequency(data_d.get("frequency", "diaria"))
        data_d["return_type"] = ReturnType(data_d.get("return_type", "simples"))
        data_d["start"] = _date(data_d.get("start"))
        data_d["end"] = _date(data_d.get("end"))

        cov_d = dict(d.get("covariance", {}))
        cov_d["method"] = CovMethod(cov_d.get("method", "ledoit_wolf"))

        pri_d = dict(d.get("prior", {}))
        pri_d["weight_source"] = WeightSource(pri_d.get("weight_source", "capitalizacion"))
        pri_d["delta_method"] = DeltaMethod(pri_d.get("delta_method", "implicito"))
        pri_d["tau_method"] = TauMethod(pri_d.get("tau_method", "manual"))

        vw_d = dict(d.get("views", {}))
        vw_d["method"] = OmegaMethod(vw_d.get("method", "idzorek"))
        vw_d["views"] = [
            ViewSpec(
                kind=ViewKind(v.get("kind", "absoluta")),
                assets=dict(v.get("assets", {})),
                q=float(v.get("q", 0.0)),
                confidence=float(v.get("confidence", 0.5)),
                omega_manual=v.get("omega_manual"),
                label=v.get("label", ""),
                enabled=bool(v.get("enabled", True)),
            )
            for v in vw_d.get("views", [])
        ]

        opt_d = dict(d.get("optimizer", {}))
        opt_d["objective"] = Objective(opt_d.get("objective", "max_sharpe"))
        opt_d["asset_bounds"] = {
            k: (float(v[0]), float(v[1])) for k, v in opt_d.get("asset_bounds", {}).items()
        }
        opt_d["groups"] = [GroupConstraint(**g) for g in opt_d.get("groups", [])]

        return cls(
            data=DataConfig(**data_d),
            covariance=CovConfig(**cov_d),
            prior=PriorConfig(**pri_d),
            views=ViewsConfig(**vw_d),
            optimizer=OptimizerConfig(**opt_d),
            version=d.get("version", "1.0.0"),
        )

    @classmethod
    def from_json(cls, s: str) -> ModelConfig:
        return cls.from_dict(json.loads(s))


# ==========================================================================
# Descarga, validación y preparación de datos de Yahoo Finance
# (originalmente blmodel/data.py)
# ==========================================================================

logger = logging.getLogger(__name__)

try:  # pragma: no cover - el import falla sólo si falta la dependencia
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


class DataError(RuntimeError):
    """Error irrecuperable en la capa de datos."""


# --------------------------------------------------------------------------
# Reporte de validación
# --------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Resultado de auditar una matriz de precios."""

    requested: list[str] = field(default_factory=list)
    accepted: list[str] = field(default_factory=list)
    invalid_tickers: list[str] = field(default_factory=list)
    dropped_short: dict[str, int] = field(default_factory=dict)
    dropped_missing: dict[str, float] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    outliers: dict[str, int] = field(default_factory=dict)
    zero_variance: list[str] = field(default_factory=list)
    start: date | None = None
    end: date | None = None
    n_observations: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.accepted) >= 2

    def to_frame(self) -> pd.DataFrame:
        """Tabla por ticker, lista para mostrar en la interfaz."""
        rows = []
        for t in self.requested:
            if t in self.invalid_tickers:
                estado, detalle = "Rechazado", "Ticker inválido o sin datos en Yahoo"
            elif t in self.dropped_short:
                estado = "Rechazado"
                detalle = f"Historia insuficiente ({self.dropped_short[t]} observaciones)"
            elif t in self.dropped_missing:
                estado = "Rechazado"
                detalle = f"Datos faltantes {self.dropped_missing[t]:.1%}"
            elif t in self.zero_variance:
                estado, detalle = "Rechazado", "Serie constante (varianza cero)"
            elif t in self.accepted:
                estado = "Aceptado"
                partes = [f"faltantes {self.missing_pct.get(t, 0.0):.1%}"]
                if self.outliers.get(t):
                    partes.append(f"{self.outliers[t]} atípicos")
                detalle = ", ".join(partes)
            else:
                estado, detalle = "Rechazado", "No disponible"
            rows.append({"Ticker": t, "Estado": estado, "Detalle": detalle})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------
def resolve_window(cfg: DataConfig) -> tuple[date, date]:
    """Resuelve la ventana de fechas efectiva."""
    end = cfg.end or date.today()
    if cfg.start is not None:
        start = cfg.start
    else:
        start = end - timedelta(days=int(round(cfg.lookback_years * 365.25)))
    if start >= end:
        raise DataError("La fecha inicial debe ser anterior a la final.")
    return start, end


def download_prices(
    tickers: list[str],
    start: date,
    end: date,
    frequency: Frequency = Frequency.DAILY,
) -> pd.DataFrame:
    """Descarga precios ajustados (dividendos y splits) de Yahoo Finance.

    Devuelve un DataFrame con una columna por ticker. Los tickers que Yahoo no
    reconoce simplemente no aparecen; `validate_prices` los reporta.
    """
    if yf is None:  # pragma: no cover
        raise DataError("yfinance no está instalado. Ejecute: pip install yfinance")
    if not tickers:
        raise DataError("No se especificó ningún ticker.")

    unique = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not unique:
        raise DataError("La lista de tickers quedó vacía tras la limpieza.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers=unique,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=frequency.yf_interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )

    if raw is None or len(raw) == 0:
        raise DataError(
            "Yahoo Finance no devolvió datos. Revise los tickers y la ventana de fechas."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise DataError("La respuesta de Yahoo Finance no contiene precios de cierre.")
        prices = raw["Close"].copy()
    else:  # un solo ticker
        if "Close" not in raw.columns:
            raise DataError("La respuesta de Yahoo Finance no contiene precios de cierre.")
        prices = raw[["Close"]].copy()
        prices.columns = unique[:1]

    prices = prices.reindex(columns=[t for t in unique if t in prices.columns])
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    return prices


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def validate_prices(
    prices: pd.DataFrame,
    requested: list[str],
    cfg: DataConfig,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Audita la matriz de precios y devuelve la versión depurada.

    Criterios de rechazo, en este orden:
      1. El ticker no existe en la respuesta de Yahoo.
      2. Menos de `min_observations` observaciones válidas.
      3. Más de `max_missing_pct` de datos faltantes dentro de la ventana común.
      4. Serie constante (varianza cero): rompe la covarianza.

    Los valores atípicos se reportan pero no se eliminan: borrar un movimiento
    extremo real sesga la covarianza a la baja. La decisión queda con el usuario.
    """
    requested = [t.strip().upper() for t in requested if t and t.strip()]
    requested = list(dict.fromkeys(requested))
    rep = ValidationReport(requested=requested)

    rep.invalid_tickers = [t for t in requested if t not in prices.columns]
    df = prices[[c for c in prices.columns if c in requested]].copy()

    if df.empty:
        rep.warnings.append("Ningún ticker devolvió datos utilizables.")
        return df, rep

    # 2. Historia suficiente
    counts = df.notna().sum()
    short = counts[counts < cfg.min_observations]
    rep.dropped_short = {t: int(n) for t, n in short.items()}
    df = df.drop(columns=list(short.index))

    if df.empty:
        rep.warnings.append(
            f"Ningún ticker alcanza el mínimo de {cfg.min_observations} observaciones."
        )
        return df, rep

    # Ventana común: desde que todas las series tienen dato
    first_valid = df.apply(lambda s: s.first_valid_index())
    common_start = max(v for v in first_valid if v is not None)
    df = df.loc[common_start:]

    # 3. Faltantes dentro de la ventana común
    miss = df.isna().mean()
    rep.missing_pct = {t: float(v) for t, v in miss.items()}
    too_many = miss[miss > cfg.max_missing_pct]
    rep.dropped_missing = {t: float(v) for t, v in too_many.items()}
    df = df.drop(columns=list(too_many.index))

    if df.empty:
        rep.warnings.append("Todos los tickers exceden el máximo de datos faltantes.")
        return df, rep

    # Relleno conservador: sólo hacia adelante, nunca hacia atrás
    # (rellenar hacia atrás introduce información del futuro).
    df = df.ffill()
    if cfg.drop_incomplete:
        df = df.dropna(how="any")

    # 4. Varianza cero
    rets = df.pct_change().dropna(how="all")
    if not rets.empty:
        const = rets.std()
        zero = const[const <= 0].index.tolist()
        rep.zero_variance = list(zero)
        df = df.drop(columns=zero)
        rets = rets.drop(columns=zero)

    # Atípicos (informativo)
    if not rets.empty and len(rets) > 2:
        z = (rets - rets.mean()) / rets.std(ddof=1).replace(0, np.nan)
        rep.outliers = {t: int((z[t].abs() > cfg.outlier_sigma).sum()) for t in rets.columns}

    rep.accepted = list(df.columns)
    rep.n_observations = int(len(df))
    if len(df):
        rep.start = df.index[0].date()
        rep.end = df.index[-1].date()

    # Advertencia de dimensionalidad: T < 2N hace inestable la covarianza muestral
    n, t = len(rep.accepted), rep.n_observations
    if n >= 2 and t < 2 * n:
        rep.warnings.append(
            f"Observaciones ({t}) menores al doble del número de activos ({n}). "
            "La covarianza muestral será inestable: use un estimador con shrinkage."
        )
    if rep.invalid_tickers:
        rep.warnings.append(
            "Tickers no reconocidos por Yahoo Finance: " + ", ".join(rep.invalid_tickers)
        )
    if len(rep.accepted) < 2:
        rep.warnings.append("Se requieren al menos dos activos válidos para optimizar.")

    return df, rep


# --------------------------------------------------------------------------
# Rendimientos
# --------------------------------------------------------------------------
def compute_returns(prices: pd.DataFrame, return_type: ReturnType) -> pd.DataFrame:
    """Rendimientos periódicos a partir de precios ajustados."""
    if return_type is ReturnType.LOG:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="any")


# --------------------------------------------------------------------------
# Capitalización de mercado y tasa libre de riesgo
# --------------------------------------------------------------------------
def fetch_market_caps(tickers: list[str]) -> tuple[pd.Series, list[str]]:
    """Capitalización de mercado por ticker.

    Yahoo no reporta capitalización para índices ni para muchos ETFs. Devuelve
    la serie con lo que sí obtuvo y la lista de faltantes, para que la capa
    superior decida el fallback en lugar de inventarlo aquí.
    """
    if yf is None:  # pragma: no cover
        raise DataError("yfinance no está instalado.")

    caps: dict[str, float] = {}
    missing: list[str] = []
    for t in tickers:
        cap = None
        try:
            tk = yf.Ticker(t)
            fast = getattr(tk, "fast_info", None)
            if fast is not None:
                cap = getattr(fast, "market_cap", None)
                if cap is None:
                    shares = getattr(fast, "shares", None)
                    price = getattr(fast, "last_price", None)
                    if shares and price:
                        cap = float(shares) * float(price)
            if not cap:
                info = tk.get_info()
                cap = info.get("marketCap") or info.get("totalAssets")
        except Exception as exc:  # pragma: no cover - depende de la red
            logger.debug("Sin capitalización para %s: %s", t, exc)
            cap = None
        if cap and float(cap) > 0:
            caps[t] = float(cap)
        else:
            missing.append(t)
    return pd.Series(caps, dtype=float), missing


def fetch_risk_free_rate(ticker: str = "^IRX", default: float = 0.04) -> tuple[float, str]:
    """Tasa libre de riesgo anual, en decimal.

    Los tickers de rendimiento de Yahoo (^IRX, ^TNX, ^FVX, ^TYX) cotizan en
    porcentaje; se dividen entre 100. Devuelve también una nota sobre la fuente
    para que quede asentada en el reporte.
    """
    if yf is None:  # pragma: no cover
        return default, "yfinance no disponible; se usó el valor por defecto."
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = yf.Ticker(ticker).history(period="1mo", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist:
            return default, f"{ticker} sin datos; se usó el valor por defecto."
        value = float(hist["Close"].dropna().iloc[-1])
        rate = value / 100.0 if ticker.startswith("^") else value
        if not np.isfinite(rate) or rate < -0.05 or rate > 0.50:
            return default, f"{ticker} devolvió {value}, fuera de rango; se usó el valor por defecto."
        return rate, f"{ticker} al {hist.index[-1].date().isoformat()}: {rate:.2%} anual."
    except Exception as exc:  # pragma: no cover - depende de la red
        return default, f"No se pudo leer {ticker} ({exc}); se usó el valor por defecto."


def annualize_factor(frequency: Frequency) -> int:
    return frequency.periods_per_year


# ==========================================================================
# Estimadores de la matriz de covarianzas
# (originalmente blmodel/covariance.py)
# ==========================================================================

@dataclass
class CovDiagnostics:
    """Diagnóstico de la matriz estimada."""

    method: str
    n_assets: int
    n_observations: int
    condition_number: float
    min_eigenvalue: float
    was_repaired: bool
    shrinkage: float | None = None

    @property
    def is_ill_conditioned(self) -> bool:
        return self.condition_number > 1e4

    def notes(self) -> list[str]:
        out: list[str] = []
        if self.n_observations < 2 * self.n_assets:
            out.append(
                f"T={self.n_observations} < 2N={2 * self.n_assets}: la covarianza "
                "muestral es inestable; se recomienda shrinkage."
            )
        if self.is_ill_conditioned:
            out.append(
                f"Número de condición {self.condition_number:,.0f}: la inversión de Σ "
                "amplifica el error de estimación."
            )
        if self.was_repaired:
            out.append(
                "Σ no era positiva definida y se reparó por proyección al cono PSD."
            )
        return out


# --------------------------------------------------------------------------
# Reparación al cono positivo definido
# --------------------------------------------------------------------------
def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> tuple[np.ndarray, bool]:
    """Proyecta una matriz simétrica al cono positivo definido.

    Descompone en valores propios, trunca los negativos a `epsilon` y
    reconstruye. Preserva los vectores propios, es decir, la estructura de
    correlación; sólo corrige la escala de las direcciones degeneradas.
    """
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    if eigvals.min() > epsilon:
        return sym, False
    clipped = np.clip(eigvals, epsilon, None)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    return repaired, True


def condition_number(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    lo = float(np.min(np.abs(eigvals)))
    hi = float(np.max(np.abs(eigvals)))
    return hi / lo if lo > 0 else float("inf")


# --------------------------------------------------------------------------
# Estimadores
# --------------------------------------------------------------------------
def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Covarianza muestral insesgada (ddof=1)."""
    return np.asarray(returns.cov(ddof=1).values, dtype=float)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Shrinkage de Ledoit-Wolf hacia una matriz objetivo de varianza constante."""
    est = LedoitWolf(assume_centered=False).fit(returns.values)
    return np.asarray(est.covariance_, dtype=float), float(est.shrinkage_)


def oas_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Oracle Approximating Shrinkage: intensidad óptima bajo normalidad."""
    est = OAS(assume_centered=False).fit(returns.values)
    return np.asarray(est.covariance_, dtype=float), float(est.shrinkage_)


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """Covarianza con ponderación exponencial (RiskMetrics).

    Pondera la observación de hace k períodos con (1-λ)λ^k, normalizado. Un λ
    menor reacciona más rápido a cambios de régimen a costa de más ruido.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda debe estar en (0, 1).")
    x = returns.values.astype(float)
    t = x.shape[0]
    if t < 2:
        raise ValueError("Se requieren al menos dos observaciones.")
    centered = x - x.mean(axis=0)
    # La observación más reciente recibe el mayor peso.
    ages = np.arange(t - 1, -1, -1, dtype=float)
    weights = (1.0 - lam) * lam**ages
    weights /= weights.sum()
    weighted = centered * weights[:, None]
    cov = weighted.T @ centered
    return 0.5 * (cov + cov.T)


def semi_covariance(returns: pd.DataFrame, threshold: float = 0.0) -> np.ndarray:
    """Semicovarianza: sólo desviaciones por debajo del umbral.

    Mide co-movimiento a la baja. Se escala por T para mantenerla comparable
    con la covarianza completa.
    """
    x = returns.values.astype(float)
    downside = np.minimum(x - threshold, 0.0)
    t = x.shape[0]
    cov = (downside.T @ downside) / t
    return 0.5 * (cov + cov.T)


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------
def estimate_covariance(
    returns: pd.DataFrame,
    cfg: CovConfig,
    periods_per_year: int,
) -> tuple[pd.DataFrame, CovDiagnostics]:
    """Estima Σ **anualizada** según la configuración.

    Devuelve la matriz con índices y columnas etiquetados, más su diagnóstico.
    """
    if returns.shape[1] < 1:
        raise ValueError("Se requiere al menos un activo.")
    if len(returns) < 2:
        raise ValueError("Se requieren al menos dos observaciones.")

    shrinkage: float | None = None
    if cfg.method is CovMethod.SAMPLE:
        cov = sample_covariance(returns)
    elif cfg.method is CovMethod.LEDOIT_WOLF:
        cov, shrinkage = ledoit_wolf_covariance(returns)
    elif cfg.method is CovMethod.OAS:
        cov, shrinkage = oas_covariance(returns)
    elif cfg.method is CovMethod.EWMA:
        cov = ewma_covariance(returns, cfg.ewma_lambda)
    elif cfg.method is CovMethod.SEMI:
        cov = semi_covariance(returns, cfg.semi_threshold)
    else:  # pragma: no cover
        raise ValueError(f"Método de covarianza no soportado: {cfg.method}")

    cov = cov * periods_per_year  # anualización

    repaired = False
    if cfg.force_psd:
        cov, repaired = ensure_psd(cov)

    diag = CovDiagnostics(
        method=cfg.method.value,
        n_assets=int(returns.shape[1]),
        n_observations=int(len(returns)),
        condition_number=condition_number(cov),
        min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),
        was_repaired=repaired,
        shrinkage=shrinkage,
    )
    frame = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return frame, diag


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlaciones implícita en Σ."""
    sd = np.sqrt(np.diag(cov.values))
    denom = np.outer(sd, sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov.values / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def annualized_volatilities(cov: pd.DataFrame) -> pd.Series:
    return pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index, name="Volatilidad")


# ==========================================================================
# Prior de equilibrio: optimización inversa
# (originalmente blmodel/prior.py)
# ==========================================================================

@dataclass
class PriorResult:
    """Prior de equilibrio y la trazabilidad de cómo se construyó."""

    pi: pd.Series
    weights: pd.Series
    delta: float
    tau: float
    risk_free: float
    weight_source: str
    delta_source: str
    tau_source: str
    rf_source: str
    fallbacks: list[str] = field(default_factory=list)
    market_caps: pd.Series | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Peso de mercado": self.weights, "Π (equilibrio)": self.pi}
        )


# --------------------------------------------------------------------------
# δ — aversión al riesgo del inversionista representativo
# --------------------------------------------------------------------------
def implied_risk_aversion(
    benchmark_returns: pd.Series,
    risk_free: float,
    periods_per_year: int,
) -> float:
    """δ = (E[R_b] − r_f) / σ²_b, con media y varianza anualizadas.

    Es la razón de Sharpe del mercado dividida entre su volatilidad: cuánto
    rendimiento excedente exige el mercado por unidad de varianza.
    """
    r = benchmark_returns.dropna()
    if len(r) < 2:
        raise ValueError("Serie del benchmark insuficiente para estimar δ.")
    mean_annual = float(r.mean()) * periods_per_year
    var_annual = float(r.var(ddof=1)) * periods_per_year
    if var_annual <= 0:
        raise ValueError("La varianza del benchmark es cero; δ no está definido.")
    return (mean_annual - risk_free) / var_annual


def sanitize_delta(delta: float, floor: float = 0.5, cap: float = 10.0) -> tuple[float, str | None]:
    """Acota δ a un rango económicamente sensato.

    δ estimado de una ventana corta puede salir negativo (mercado a la baja) o
    absurdamente grande. Un δ ≤ 0 invierte el signo de todo el prior, así que se
    acota y se avisa en lugar de propagar el disparate.
    """
    if not np.isfinite(delta):
        return 2.5, "δ estimado no finito; se fijó en 2.5."
    if delta < floor:
        return floor, (
            f"δ estimado en {delta:.2f} (el benchmark rindió por debajo de la tasa "
            f"libre de riesgo en la ventana). Se acotó a {floor:.2f}."
        )
    if delta > cap:
        return cap, f"δ estimado en {delta:.2f}; se acotó a {cap:.2f}."
    return float(delta), None


# --------------------------------------------------------------------------
# w_mkt — pesos del portafolio de equilibrio
# --------------------------------------------------------------------------
def weights_from_market_caps(caps: pd.Series, assets: list[str]) -> pd.Series:
    """Normaliza capitalizaciones al universo analizado.

    Nota metodológica: al restringir el universo a unos pocos activos, estos
    pesos ya no son los del mercado global sino los del *subuniverso*. Es la
    aproximación estándar y hay que declararla.
    """
    sub = caps.reindex(assets).dropna()
    if sub.empty or sub.sum() <= 0:
        raise ValueError("No hay capitalizaciones válidas para normalizar.")
    return (sub / sub.sum()).reindex(assets).fillna(0.0)


def equal_weights(assets: list[str]) -> pd.Series:
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=assets)


def normalize_weights(raw: dict[str, float], assets: list[str]) -> pd.Series:
    s = pd.Series({a: float(raw.get(a, 0.0)) for a in assets}, dtype=float)
    total = s.sum()
    if abs(total) < 1e-12:
        raise ValueError("Los pesos propios suman cero.")
    return s / total


def resolve_market_weights(
    assets: list[str],
    cfg: PriorConfig,
    market_caps: pd.Series | None = None,
    benchmark_weights: pd.Series | None = None,
) -> tuple[pd.Series, str, list[str]]:
    """Resuelve w_mkt según la fuente elegida, con cascada de respaldo.

    Cascada: fuente elegida → capitalización → equiponderado. Cada salto queda
    asentado en la lista de fallbacks, porque cambia la interpretación del prior.
    """
    fallbacks: list[str] = []

    if cfg.weight_source is WeightSource.CUSTOM:
        try:
            return normalize_weights(cfg.custom_weights, assets), "Pesos propios", fallbacks
        except ValueError:
            fallbacks.append("Pesos propios inválidos; se usó capitalización de mercado.")

    if cfg.weight_source is WeightSource.BENCHMARK:
        if benchmark_weights is not None and not benchmark_weights.empty:
            sub = benchmark_weights.reindex(assets).dropna()
            if not sub.empty and sub.sum() > 0:
                w = (sub / sub.sum()).reindex(assets).fillna(0.0)
                return w, "Pesos del benchmark", fallbacks
        fallbacks.append(
            "No hay pesos del benchmark disponibles; se usó capitalización de mercado."
        )

    if cfg.weight_source is WeightSource.EQUAL:
        return equal_weights(assets), "Equiponderado", fallbacks

    # Capitalización de mercado (elección directa o destino de la cascada)
    if market_caps is not None:
        available = market_caps.reindex(assets).dropna()
        coverage = len(available) / max(len(assets), 1)
        if coverage >= 0.999:
            return weights_from_market_caps(market_caps, assets), "Capitalización de mercado", fallbacks
        if coverage > 0:
            missing = [a for a in assets if a not in available.index]
            fallbacks.append(
                "Yahoo Finance no reporta capitalización para "
                f"{', '.join(missing)}. Se usó equiponderado para todo el universo "
                "en lugar de mezclar dos criterios de peso."
            )
        else:
            fallbacks.append(
                "Yahoo Finance no reporta capitalización para ningún activo "
                "(típico en índices y algunos ETFs). Se usó equiponderado."
            )
    else:
        fallbacks.append("No se consultó capitalización de mercado; se usó equiponderado.")

    return equal_weights(assets), "Equiponderado (respaldo)", fallbacks


# --------------------------------------------------------------------------
# τ — incertidumbre del prior
# --------------------------------------------------------------------------
def resolve_tau(cfg: PriorConfig, n_observations: int) -> tuple[float, str]:
    """τ escala la incertidumbre del prior: la distribución de la media es N(Π, τΣ).

    Dos convenciones:
      - τ = 0.05, el valor de la literatura original (He-Litterman).
      - τ = 1/T (Meucci), que hace explícito que la incertidumbre sobre la media
        cae con el tamaño de la muestra.
    """
    if cfg.tau_method is TauMethod.ONE_OVER_T:
        if n_observations <= 0:
            return 0.05, "τ = 0.05 (sin observaciones para calcular 1/T)."
        tau = 1.0 / n_observations
        return tau, f"τ = 1/T = 1/{n_observations} = {tau:.5f} (Meucci)."
    tau = float(cfg.tau_manual)
    if not 0 < tau <= 1:
        return 0.05, f"τ = {tau} fuera de (0, 1]; se fijó en 0.05."
    return tau, f"τ = {tau:.4f} (fijado por el usuario)."


# --------------------------------------------------------------------------
# Π — rendimientos de equilibrio
# --------------------------------------------------------------------------
def implied_equilibrium_returns(
    delta: float,
    cov: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Π = δ · Σ · w_mkt."""
    w = weights.reindex(cov.index).fillna(0.0).values.astype(float)
    pi = delta * (cov.values @ w)
    return pd.Series(pi, index=cov.index, name="Pi")


def build_prior(
    cov: pd.DataFrame,
    cfg: PriorConfig,
    benchmark_returns: pd.Series | None,
    periods_per_year: int,
    n_observations: int,
    risk_free: float,
    rf_source: str,
    market_caps: pd.Series | None = None,
    benchmark_weights: pd.Series | None = None,
) -> PriorResult:
    """Ensambla el prior de equilibrio completo."""
    assets = list(cov.index)
    weights, weight_source, fallbacks = resolve_market_weights(
        assets, cfg, market_caps, benchmark_weights
    )

    if cfg.delta_method is DeltaMethod.IMPLIED and benchmark_returns is not None:
        try:
            raw_delta = implied_risk_aversion(benchmark_returns, risk_free, periods_per_year)
            delta, note = sanitize_delta(raw_delta)
            delta_source = f"δ implícito del benchmark = {delta:.3f}"
            if note:
                fallbacks.append(note)
        except ValueError as exc:
            delta = float(cfg.delta_manual)
            delta_source = f"δ = {delta:.3f} (manual)"
            fallbacks.append(f"No se pudo estimar δ ({exc}); se usó el valor manual.")
    else:
        delta = float(cfg.delta_manual)
        delta_source = f"δ = {delta:.3f} (fijado por el usuario)"
        if cfg.delta_method is DeltaMethod.IMPLIED:
            fallbacks.append("Sin serie del benchmark; δ tomó el valor manual.")

    tau, tau_source = resolve_tau(cfg, n_observations)
    pi = implied_equilibrium_returns(delta, cov, weights)

    return PriorResult(
        pi=pi,
        weights=weights.reindex(assets).fillna(0.0),
        delta=float(delta),
        tau=float(tau),
        risk_free=float(risk_free),
        weight_source=weight_source,
        delta_source=delta_source,
        tau_source=tau_source,
        rf_source=rf_source,
        fallbacks=fallbacks,
        market_caps=market_caps,
    )


# ==========================================================================
# Views del inversionista: construcción de P, Q y Ω
# (originalmente blmodel/views.py)
# ==========================================================================

@dataclass
class ViewMatrices:
    """P, Q y Ω listas para el motor, más la trazabilidad de su construcción."""

    P: np.ndarray
    Q: np.ndarray
    omega: np.ndarray
    assets: list[str]
    labels: list[str]
    confidences: list[float]
    method: str
    warnings: list[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return int(self.P.shape[0])

    def p_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.P, index=self.labels, columns=self.assets)

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "View": self.labels,
                "Q (rendimiento)": self.Q,
                "Confianza": self.confidences,
                "Ω (varianza del error)": np.diag(self.omega),
            }
        )


# --------------------------------------------------------------------------
# Validación y construcción de P, Q
# --------------------------------------------------------------------------
def validate_view(view: ViewSpec, assets: list[str]) -> list[str]:
    """Revisa una view contra el universo. Devuelve la lista de problemas."""
    problems: list[str] = []
    if not view.assets:
        problems.append("La view no especifica activos.")
        return problems

    unknown = [a for a in view.assets if a not in assets]
    if unknown:
        problems.append(f"Activos fuera del universo: {', '.join(unknown)}.")

    total = sum(view.assets.values())
    if view.kind is ViewKind.RELATIVE:
        if abs(total) > 1e-6:
            problems.append(
                f"Una view relativa debe tener pesos que sumen cero (suman {total:+.4f})."
            )
        if len(view.assets) < 2:
            problems.append("Una view relativa requiere al menos dos activos.")
    else:
        if abs(total - 1.0) > 1e-6:
            problems.append(
                f"Una view absoluta debe tener pesos que sumen uno (suman {total:.4f})."
            )

    if not np.isfinite(view.q):
        problems.append("El rendimiento Q de la view no es finito.")
    if not 0.0 <= view.confidence <= 1.0:
        problems.append("La confianza debe estar entre 0% y 100%.")
    return problems


def build_p_q(views: list[ViewSpec], assets: list[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[float]]:
    """Construye P (k×n) y Q (k,) a partir de las views activas y válidas."""
    rows, qs, labels, confs = [], [], [], []
    index = {a: i for i, a in enumerate(assets)}

    for i, v in enumerate(views):
        if not v.enabled:
            continue
        if validate_view(v, assets):
            continue
        row = np.zeros(len(assets), dtype=float)
        for asset, weight in v.assets.items():
            row[index[asset]] = float(weight)
        rows.append(row)
        qs.append(float(v.q))
        labels.append(v.label or _default_label(v, i))
        confs.append(float(v.confidence))

    if not rows:
        return np.zeros((0, len(assets))), np.zeros(0), [], []
    return np.vstack(rows), np.asarray(qs, dtype=float), labels, confs


def _default_label(view: ViewSpec, i: int) -> str:
    if view.kind is ViewKind.ABSOLUTE:
        asset = next(iter(view.assets), "?")
        return f"{asset} rinde {view.q:.2%}"
    pos = [a for a, w in view.assets.items() if w > 0]
    neg = [a for a, w in view.assets.items() if w < 0]
    return f"{'+'.join(pos) or '?'} supera a {'+'.join(neg) or '?'} por {view.q:.2%}"


# --------------------------------------------------------------------------
# Ω — método He-Litterman (proporcional)
# --------------------------------------------------------------------------
def omega_he_litterman(P: np.ndarray, tau: float, cov: np.ndarray) -> np.ndarray:
    """Ω = diag(P · τΣ · Pᵀ).

    Cada view hereda la incertidumbre que el prior ya tiene en esa dirección.
    No hay parámetro subjetivo: la confianza relativa entre views queda fijada
    por la estructura de Σ, no por el usuario.
    """
    inner = P @ (tau * cov) @ P.T
    diag = np.clip(np.diag(inner), 1e-12, None)
    return np.diag(diag)


# --------------------------------------------------------------------------
# Ω — método Idzorek (confianza 0–100%)
# --------------------------------------------------------------------------
def _posterior_mu(
    pi: np.ndarray,
    tau: float,
    cov: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Forma estable del posterior: μ = Π + τΣPᵀ(PτΣPᵀ + Ω)⁻¹(Q − PΠ).

    Invierte una matriz k×k (número de views) en lugar de una n×n, que es tanto
    más barato como numéricamente mejor cuando hay pocas views.
    """
    tau_cov = tau * cov
    middle = P @ tau_cov @ P.T + omega
    adjustment = tau_cov @ P.T @ np.linalg.solve(middle, Q - P @ pi)
    return pi + adjustment


def _implied_weights(mu: np.ndarray, delta: float, cov: np.ndarray) -> np.ndarray:
    """w = (δΣ)⁻¹ μ — el portafolio sin restricciones que implica μ."""
    return np.linalg.solve(delta * cov, mu)


def omega_idzorek(
    P: np.ndarray,
    Q: np.ndarray,
    confidences: list[float],
    pi: np.ndarray,
    tau: float,
    cov: np.ndarray,
    delta: float,
    w_market: np.ndarray,
) -> np.ndarray:
    """Ω calibrada por el método de Idzorek (2005).

    El usuario declara confianza c ∈ [0, 1]. El anclaje es:

      c = 0   →  el portafolio resultante es el de mercado (la view se ignora)
      c = 1   →  el portafolio resultante es el de confianza total
      c = ½   →  el peso se mueve la mitad de la distancia entre ambos

    Para cada view se calcula el portafolio de confianza total, se define el
    objetivo w_obj = w_mkt + c·(w_100 − w_mkt) y se resuelve numéricamente el
    escalar ω que reproduce ese objetivo. Es una calibración pragmática, no un
    resultado bayesiano cerrado: por eso el método He-Litterman queda disponible
    como contraste.
    """
    k, n = P.shape
    omegas = np.zeros(k, dtype=float)
    tau_cov = tau * cov

    for i in range(k):
        Pi_row = P[i : i + 1, :]
        Qi = Q[i : i + 1]
        conf = float(np.clip(confidences[i], 0.0, 1.0))

        base = float((Pi_row @ tau_cov @ Pi_row.T).item())
        base = max(base, 1e-12)

        # Confianza nula: la view no debe mover nada.
        if conf <= 1e-9:
            omegas[i] = base * 1e9
            continue
        # Confianza total: Ω → 0, acotado para no romper la inversión.
        if conf >= 1.0 - 1e-9:
            omegas[i] = base * 1e-9
            continue

        # Portafolio con la view impuesta como certeza
        mu_100 = _posterior_mu(pi, tau, cov, Pi_row, Qi, np.array([[base * 1e-10]]))
        w_100 = _implied_weights(mu_100, delta, cov)
        target = w_market + conf * (w_100 - w_market)

        def objective(log_omega: float, _Pi=Pi_row, _Qi=Qi, _target=target) -> float:
            omega = np.array([[np.exp(log_omega)]])
            mu = _posterior_mu(pi, tau, cov, _Pi, _Qi, omega)
            w = _implied_weights(mu, delta, cov)
            return float(np.sum((w - _target) ** 2))

        # Se busca en log(ω) porque ω recorre varios órdenes de magnitud.
        lo, hi = np.log(base) - 25.0, np.log(base) + 25.0
        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-8})
        omegas[i] = float(np.exp(res.x)) if res.success else base

    return np.diag(np.clip(omegas, 1e-12, None))


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------
def build_view_matrices(
    cfg: ViewsConfig,
    assets: list[str],
    pi: pd.Series,
    cov: pd.DataFrame,
    tau: float,
    delta: float,
    w_market: pd.Series,
) -> ViewMatrices:
    """Construye P, Q y Ω según el método configurado."""
    warnings_out: list[str] = []
    for i, v in enumerate(cfg.views):
        if not v.enabled:
            continue
        for problem in validate_view(v, assets):
            warnings_out.append(f"View {i + 1} descartada: {problem}")

    P, Q, labels, confs = build_p_q(cfg.views, assets)
    cov_np = np.asarray(cov.values, dtype=float)

    if P.shape[0] == 0:
        return ViewMatrices(
            P=P, Q=Q, omega=np.zeros((0, 0)), assets=assets, labels=[],
            confidences=[], method="sin views", warnings=warnings_out,
        )

    pi_np = np.asarray(pi.reindex(assets).values, dtype=float)
    w_np = np.asarray(w_market.reindex(assets).fillna(0.0).values, dtype=float)

    if cfg.method is OmegaMethod.HE_LITTERMAN:
        omega = omega_he_litterman(P, tau, cov_np)
        method = "He-Litterman proporcional"
    elif cfg.method is OmegaMethod.MANUAL:
        manual = [v.omega_manual for v in cfg.views if v.enabled and not validate_view(v, assets)]
        vals = []
        for i, m in enumerate(manual):
            if m is None or not np.isfinite(m) or m <= 0:
                fallback = float(np.asarray(P[i] @ (tau * cov_np) @ P[i].T).item())
                vals.append(max(fallback, 1e-12))
                warnings_out.append(
                    f"View {i + 1} sin Ω manual válida; se usó el valor proporcional."
                )
            else:
                vals.append(float(m))
        omega = np.diag(vals)
        method = "Ω manual"
    else:
        omega = omega_idzorek(P, Q, confs, pi_np, tau, cov_np, delta, w_np)
        method = "Idzorek (confianza 0–100%)"

    return ViewMatrices(
        P=P, Q=Q, omega=omega, assets=assets, labels=labels,
        confidences=confs, method=method, warnings=warnings_out,
    )


# ==========================================================================
# Motor Black-Litterman: combinación bayesiana
# (originalmente blmodel/blacklitterman.py)
# ==========================================================================

@dataclass
class BLResult:
    """Salida del motor, con todo lo necesario para auditar el paso."""

    mu_bl: pd.Series
    cov_bl: pd.DataFrame
    pi: pd.Series
    cov_prior: pd.DataFrame
    tau: float
    delta: float
    view_impact: pd.Series          # μ_BL − Π, por activo
    view_residual: pd.Series | None  # Q − PΠ, por view
    view_realized: pd.Series | None  # P·μ_BL, por view
    n_views: int
    method: str

    def comparison_frame(self) -> pd.DataFrame:
        """Π vs μ_BL vs el efecto de las views: la tabla que explica el modelo."""
        return pd.DataFrame(
            {
                "Π (equilibrio)": self.pi,
                "μ_BL (posterior)": self.mu_bl,
                "Δ por views": self.view_impact,
            }
        )

    @property
    def total_view_impact(self) -> float:
        """Magnitud agregada del desvío que las views imprimieron sobre Π."""
        return float(np.abs(self.view_impact.values).sum())


def posterior_returns(
    pi: np.ndarray,
    cov: np.ndarray,
    tau: float,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """μ_BL = Π + τΣPᵀ (PτΣPᵀ + Ω)⁻¹ (Q − PΠ)."""
    if P.shape[0] == 0:
        return pi.copy()
    tau_cov = tau * cov
    middle = P @ tau_cov @ P.T + omega
    residual = Q - P @ pi
    return pi + tau_cov @ P.T @ np.linalg.solve(middle, residual)


def posterior_covariance(
    cov: np.ndarray,
    tau: float,
    P: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Σ_BL = Σ + M, con M = τΣ − τΣPᵀ(PτΣPᵀ + Ω)⁻¹PτΣ.

    M es la incertidumbre que queda sobre la *media* después de incorporar las
    views. Sumarla a Σ reconoce que el riesgo del portafolio incluye el error
    de estimación del propio vector de rendimientos, no sólo la volatilidad de
    los activos. Ignorar M (usar Σ a secas) es la simplificación común; aquí se
    incluye porque es lo correcto y porque la diferencia sí mueve los pesos.
    """
    tau_cov = tau * cov
    if P.shape[0] == 0:
        return cov + tau_cov
    middle = P @ tau_cov @ P.T + omega
    m = tau_cov - tau_cov @ P.T @ np.linalg.solve(middle, P @ tau_cov)
    out = cov + m
    return 0.5 * (out + out.T)


def run_black_litterman(
    pi: pd.Series,
    cov: pd.DataFrame,
    tau: float,
    delta: float,
    views: ViewMatrices,
    include_estimation_risk: bool = True,
) -> BLResult:
    """Ejecuta el paso bayesiano completo.

    Si `include_estimation_risk` es False, el posterior usa Σ tal cual en lugar
    de Σ + M. Se deja como opción porque buena parte de la literatura aplicada
    lo hace así, y conviene poder reproducir ambos resultados.
    """
    assets = list(cov.index)
    pi_np = np.asarray(pi.reindex(assets).values, dtype=float)
    cov_np = np.asarray(cov.values, dtype=float)

    mu = posterior_returns(pi_np, cov_np, tau, views.P, views.Q, views.omega)
    cov_post = (
        posterior_covariance(cov_np, tau, views.P, views.omega)
        if include_estimation_risk
        else cov_np.copy()
    )

    mu_s = pd.Series(mu, index=assets, name="mu_BL")
    impact = pd.Series(mu - pi_np, index=assets, name="Impacto de views")

    residual = realized = None
    if views.n_views > 0:
        residual = pd.Series(
            views.Q - views.P @ pi_np, index=views.labels, name="Q - P·Pi"
        )
        realized = pd.Series(views.P @ mu, index=views.labels, name="P·mu_BL")

    return BLResult(
        mu_bl=mu_s,
        cov_bl=pd.DataFrame(cov_post, index=assets, columns=assets),
        pi=pi.reindex(assets),
        cov_prior=cov,
        tau=float(tau),
        delta=float(delta),
        view_impact=impact,
        view_residual=residual,
        view_realized=realized,
        n_views=views.n_views,
        method=views.method,
    )


def view_diagnostics(result: BLResult, views: ViewMatrices) -> pd.DataFrame | None:
    """Tabla por view: lo que usted dijo, lo que decía el mercado, dónde quedó.

    La columna "Absorción" es el porcentaje del desacuerdo con el equilibrio que
    el modelo terminó incorporando. Con confianza 100% tiende a 100%; con
    confianza 0%, a 0%. Es la lectura directa de si el modelo le hizo caso.
    """
    if views.n_views == 0:
        return None
    pi_np = np.asarray(result.pi.values, dtype=float)
    mu_np = np.asarray(result.mu_bl.values, dtype=float)

    p_pi = views.P @ pi_np
    p_mu = views.P @ mu_np
    gap = views.Q - p_pi
    with np.errstate(divide="ignore", invalid="ignore"):
        absorption = np.where(np.abs(gap) > 1e-12, (p_mu - p_pi) / gap, np.nan)

    return pd.DataFrame(
        {
            "View": views.labels,
            "Q (su view)": views.Q,
            "P·Π (equilibrio)": p_pi,
            "P·μ_BL (posterior)": p_mu,
            "Desacuerdo (Q − P·Π)": gap,
            "Absorción": absorption,
            "Confianza declarada": views.confidences,
            "Ω": np.diag(views.omega),
        }
    )


def tau_sensitivity(
    pi: pd.Series,
    cov: pd.DataFrame,
    delta: float,
    views: ViewMatrices,
    taus: np.ndarray,
    weight_fn=None,
) -> pd.DataFrame:
    """Recorre τ y devuelve los pesos implícitos resultantes.

    τ es el parámetro más discutido del modelo y el menos observable. Esta
    tabla convierte esa discusión en algo que se mira: si los pesos se mueven
    poco en el rango razonable de τ, la elección deja de importar; si se mueven
    mucho, hay que justificarla.
    """
    assets = list(cov.index)
    cov_np = np.asarray(cov.values, dtype=float)
    pi_np = np.asarray(pi.reindex(assets).values, dtype=float)

    rows = []
    for tau in taus:
        mu = posterior_returns(pi_np, cov_np, float(tau), views.P, views.Q, views.omega)
        if weight_fn is not None:
            w = weight_fn(pd.Series(mu, index=assets), float(tau))
            w = np.asarray(pd.Series(w).reindex(assets).fillna(0.0).values, dtype=float)
        else:
            w = np.linalg.solve(delta * cov_np, mu)
        rows.append(pd.Series(w, index=assets, name=float(tau)))
    return pd.DataFrame(rows)


# ==========================================================================
# Optimización de portafolio con restricciones (cvxpy)
# (originalmente blmodel/optimizer.py)
# ==========================================================================

class OptimizationError(RuntimeError):
    """El problema resultó infactible o el solver no convergió."""


@dataclass
class PortfolioResult:
    """Portafolio óptimo y sus métricas ex-ante."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    objective: str
    status: str
    diagnostics: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return self.weights.rename("Peso").to_frame()


# --------------------------------------------------------------------------
# Construcción de restricciones
# --------------------------------------------------------------------------
def _bounds(assets: list[str], cfg: OptimizerConfig) -> tuple[np.ndarray, np.ndarray]:
    """Cotas por activo, combinando la cota global con las específicas."""
    lo_default = cfg.weight_lower if cfg.allow_short else max(cfg.weight_lower, 0.0)
    lo = np.full(len(assets), float(lo_default))
    hi = np.full(len(assets), float(cfg.weight_upper))
    for i, a in enumerate(assets):
        if a in cfg.asset_bounds:
            low, high = cfg.asset_bounds[a]
            lo[i], hi[i] = float(low), float(high)
    if np.any(lo > hi):
        bad = [assets[i] for i in np.where(lo > hi)[0]]
        raise OptimizationError(
            f"Cotas inconsistentes (mínimo mayor que máximo) en: {', '.join(bad)}."
        )
    if lo.sum() > 1.0 + 1e-9:
        raise OptimizationError(
            f"Los pesos mínimos suman {lo.sum():.2%}, más del 100%. El problema es infactible."
        )
    if hi.sum() < 1.0 - 1e-9:
        raise OptimizationError(
            f"Los pesos máximos suman {hi.sum():.2%}, menos del 100%. El problema es infactible."
        )
    return lo, hi


def _group_masks(assets: list[str], cfg: OptimizerConfig) -> list[tuple[np.ndarray, float, float, str]]:
    out = []
    index = {a: i for i, a in enumerate(assets)}
    for g in cfg.groups:
        mask = np.zeros(len(assets))
        for m in g.members:
            if m in index:
                mask[index[m]] = 1.0
        if mask.sum() == 0:
            continue
        out.append((mask, float(g.lower), float(g.upper), g.name))
    return out


def _base_constraints(w: cp.Variable, assets: list[str], cfg: OptimizerConfig) -> list:
    """Restricciones en el espacio original de pesos."""
    lo, hi = _bounds(assets, cfg)
    cons = [cp.sum(w) == 1, w >= lo, w <= hi]
    for mask, gl, gu, _ in _group_masks(assets, cfg):
        cons += [mask @ w >= gl, mask @ w <= gu]
    if cfg.allow_short and cfg.max_gross_leverage > 0:
        cons.append(cp.norm1(w) <= cfg.max_gross_leverage)
    return cons


def _homogeneous_constraints(
    y: cp.Variable, kappa: cp.Variable, assets: list[str], cfg: OptimizerConfig
) -> list:
    """Las mismas restricciones tras la sustitución w = y/κ.

    Cada restricción lineal a'w ≤ b se vuelve a'y ≤ b·κ, que sigue siendo
    lineal. Por eso la transformación de Sharpe conserva todo el conjunto
    factible en lugar de obligar a resolver el caso irrestricto.
    """
    lo, hi = _bounds(assets, cfg)
    cons = [cp.sum(y) == kappa, kappa >= 1e-8, y >= cp.multiply(lo, kappa), y <= cp.multiply(hi, kappa)]
    for mask, gl, gu, _ in _group_masks(assets, cfg):
        cons += [mask @ y >= gl * kappa, mask @ y <= gu * kappa]
    if cfg.allow_short and cfg.max_gross_leverage > 0:
        cons.append(cp.norm1(y) <= cfg.max_gross_leverage * kappa)
    return cons


def _solve(problem: cp.Problem, label: str) -> str:
    """Resuelve probando varios solvers antes de rendirse."""
    last_error = None
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS, cp.ECOS):
        try:
            problem.solve(solver=solver)
        except Exception as exc:  # el solver puede no estar instalado
            last_error = exc
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return problem.status
    raise OptimizationError(
        f"No se pudo resolver el problema de {label} "
        f"(estado: {problem.status}; último error: {last_error}). "
        "Revise que las restricciones sean compatibles entre sí."
    )


# --------------------------------------------------------------------------
# Objetivos
# --------------------------------------------------------------------------
def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, rf: float, cfg: OptimizerConfig
) -> tuple[pd.Series, str]:
    """Máxima razón de Sharpe por transformación homogénea.

    minimizar  yᵀΣy   sujeto a   (μ − r_f)ᵀy = 1,  restricciones en (y, κ)
    y luego    w = y/κ

    Requiere que exista al menos un portafolio factible con exceso de
    rendimiento positivo; si no, el conjunto es vacío y se avisa explícitamente.
    """
    assets = list(mu.index)
    excess = mu.values - rf
    if np.max(excess) <= 0:
        raise OptimizationError(
            "Ningún activo supera la tasa libre de riesgo con los rendimientos "
            f"posteriores ({rf:.2%}). La razón de Sharpe no tiene máximo bien "
            "definido: use mínima varianza o baje la tasa libre de riesgo."
        )
    n = len(assets)
    y = cp.Variable(n)
    kappa = cp.Variable()
    cons = _homogeneous_constraints(y, kappa, assets, cfg)
    cons.append(excess @ y == 1)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov.values))), cons)
    status = _solve(prob, "máximo Sharpe")
    if y.value is None or kappa.value is None or kappa.value <= 0:
        raise OptimizationError("La transformación de Sharpe no produjo una solución válida.")
    return pd.Series(np.asarray(y.value) / float(kappa.value), index=assets), status


def min_variance(cov: pd.DataFrame, cfg: OptimizerConfig) -> tuple[pd.Series, str]:
    """Mínima varianza global: el único objetivo que no depende de μ."""
    assets = list(cov.index)
    w = cp.Variable(len(assets))
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))),
        _base_constraints(w, assets, cfg),
    )
    status = _solve(prob, "mínima varianza")
    return pd.Series(np.asarray(w.value), index=assets), status


def max_utility(
    mu: pd.Series, cov: pd.DataFrame, delta: float, cfg: OptimizerConfig
) -> tuple[pd.Series, str]:
    """maximizar wᵀμ − (δ/2)·wᵀΣw — el cierre natural de Black-Litterman."""
    assets = list(mu.index)
    w = cp.Variable(len(assets))
    utility = mu.values @ w - (delta / 2.0) * cp.quad_form(w, cp.psd_wrap(cov.values))
    prob = cp.Problem(cp.Maximize(utility), _base_constraints(w, assets, cfg))
    status = _solve(prob, "máxima utilidad")
    return pd.Series(np.asarray(w.value), index=assets), status


def target_volatility(
    mu: pd.Series, cov: pd.DataFrame, target: float, cfg: OptimizerConfig
) -> tuple[pd.Series, str]:
    """Máximo rendimiento sujeto a σ(w) ≤ objetivo (restricción de cono SOC)."""
    assets = list(mu.index)
    w = cp.Variable(len(assets))
    cons = _base_constraints(w, assets, cfg)
    cons.append(cp.quad_form(w, cp.psd_wrap(cov.values)) <= target**2)
    prob = cp.Problem(cp.Maximize(mu.values @ w), cons)
    try:
        status = _solve(prob, "volatilidad objetivo")
    except OptimizationError as exc:
        gmv, _ = min_variance(cov, cfg)
        floor = float(np.sqrt(gmv.values @ cov.values @ gmv.values))
        raise OptimizationError(
            f"Volatilidad objetivo de {target:.2%} infactible: el portafolio de "
            f"mínima varianza ya tiene {floor:.2%}. Suba el objetivo."
        ) from exc
    return pd.Series(np.asarray(w.value), index=assets), status


def target_return(
    mu: pd.Series, cov: pd.DataFrame, target: float, cfg: OptimizerConfig
) -> tuple[pd.Series, str]:
    """Mínima varianza sujeto a wᵀμ ≥ objetivo."""
    assets = list(mu.index)
    w = cp.Variable(len(assets))
    cons = _base_constraints(w, assets, cfg)
    cons.append(mu.values @ w >= target)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov.values))), cons)
    try:
        status = _solve(prob, "rendimiento objetivo")
    except OptimizationError as exc:
        raise OptimizationError(
            f"Rendimiento objetivo de {target:.2%} infactible bajo las "
            f"restricciones actuales (máximo posible ≈ {float(np.max(mu.values)):.2%})."
        ) from exc
    return pd.Series(np.asarray(w.value), index=assets), status


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------
def optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    cfg: OptimizerConfig,
    risk_free: float = 0.0,
    delta: float = 2.5,
) -> PortfolioResult:
    """Resuelve el objetivo configurado y devuelve el portafolio con métricas."""
    assets = list(cov.index)
    mu = mu.reindex(assets)
    diagnostics: list[str] = []

    if cfg.objective is Objective.MAX_SHARPE:
        w, status = max_sharpe(mu, cov, risk_free, cfg)
        label = "Máxima razón de Sharpe"
    elif cfg.objective is Objective.MIN_VARIANCE:
        w, status = min_variance(cov, cfg)
        label = "Mínima varianza"
    elif cfg.objective is Objective.MAX_UTILITY:
        w, status = max_utility(mu, cov, delta, cfg)
        label = f"Máxima utilidad (δ = {delta:.2f})"
    elif cfg.objective is Objective.TARGET_VOL:
        w, status = target_volatility(mu, cov, cfg.target_vol, cfg)
        label = f"Volatilidad objetivo {cfg.target_vol:.1%}"
    elif cfg.objective is Objective.TARGET_RETURN:
        w, status = target_return(mu, cov, cfg.target_return, cfg)
        label = f"Rendimiento objetivo {cfg.target_return:.1%}"
    else:  # pragma: no cover
        raise OptimizationError(f"Objetivo no soportado: {cfg.objective}")

    # Limpieza numérica: el solver deja residuos del orden de 1e-12
    w = w.copy()
    w[np.abs(w.values) < 1e-8] = 0.0
    total = w.sum()
    if abs(total - 1.0) > 1e-6 and abs(total) > 1e-9:
        diagnostics.append(f"Los pesos sumaban {total:.6f}; se renormalizaron a 1.")
        w = w / total

    if status == "optimal_inaccurate":
        diagnostics.append(
            "El solver reportó solución aproximada: los pesos son utilizables pero "
            "conviene revisar si las restricciones están muy ajustadas."
        )

    ret = float(w.values @ mu.values)
    vol = float(np.sqrt(max(w.values @ cov.values @ w.values, 0.0)))
    sharpe = (ret - risk_free) / vol if vol > 0 else float("nan")

    return PortfolioResult(
        weights=w, expected_return=ret, volatility=vol, sharpe=sharpe,
        objective=label, status=status, diagnostics=diagnostics,
    )


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    cfg: OptimizerConfig,
    n_points: int | None = None,
) -> pd.DataFrame:
    """Frontera eficiente bajo las mismas restricciones del portafolio óptimo.

    Barre rendimientos objetivo entre el del portafolio de mínima varianza y el
    máximo alcanzable, y resuelve mínima varianza en cada punto. Sólo se grafica
    la rama eficiente (de mínima varianza hacia arriba).
    """
    assets = list(cov.index)
    mu = mu.reindex(assets)
    n = n_points or cfg.frontier_points

    gmv, _ = min_variance(cov, cfg)
    r_min = float(gmv.values @ mu.values)

    w_max = cp.Variable(len(assets))
    prob = cp.Problem(cp.Maximize(mu.values @ w_max), _base_constraints(w_max, assets, cfg))
    _solve(prob, "rendimiento máximo")
    r_max = float(mu.values @ np.asarray(w_max.value))

    if r_max <= r_min + 1e-9:
        vol = float(np.sqrt(max(gmv.values @ cov.values @ gmv.values, 0.0)))
        return pd.DataFrame({"Rendimiento": [r_min], "Volatilidad": [vol]})

    rows = []
    for target in np.linspace(r_min, r_max, n):
        try:
            w, _ = target_return(mu, cov, float(target), cfg)
        except OptimizationError:
            continue
        vol = float(np.sqrt(max(w.values @ cov.values @ w.values, 0.0)))
        rows.append({"Rendimiento": float(w.values @ mu.values), "Volatilidad": vol})

    frontier = pd.DataFrame(rows).drop_duplicates()
    return frontier.sort_values("Volatilidad").reset_index(drop=True)


def unconstrained_weights(mu: pd.Series, cov: pd.DataFrame, delta: float) -> pd.Series:
    """w = (δΣ)⁻¹μ — la solución cerrada del paper original, sin restricciones.

    Se incluye como referencia: permite ver cuánto cuesta, en términos del
    óptimo teórico, imponer long-only y las cotas.
    """
    w = np.linalg.solve(delta * cov.values, mu.reindex(cov.index).values)
    return pd.Series(w, index=cov.index)


# ==========================================================================
# Métricas ex-ante y descomposición del riesgo
# (originalmente blmodel/metrics.py)
# ==========================================================================

def portfolio_return(weights: pd.Series, mu: pd.Series) -> float:
    w = weights.reindex(mu.index).fillna(0.0).values
    return float(w @ mu.values)


def portfolio_volatility(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).fillna(0.0).values
    return float(np.sqrt(max(w @ cov.values @ w, 0.0)))


def sharpe_ratio(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol <= 0:
        return float("nan")
    return (portfolio_return(weights, mu) - rf) / vol


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Descomposición de Euler del riesgo total.

    CR_i = w_i · (Σw)_i / σ_p. Suman exactamente σ_p, así que el porcentaje
    responde "¿de dónde viene el riesgo?" — que casi nunca coincide con "¿dónde
    está el dinero?".
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    sigma_w = cov.values @ w
    vol = float(np.sqrt(max(w @ sigma_w, 0.0)))
    if vol <= 0:
        zeros = np.zeros(len(w))
        return pd.DataFrame(
            {"Peso": w, "Contribución marginal": zeros,
             "Contribución al riesgo": zeros, "% del riesgo": zeros},
            index=cov.index,
        )
    marginal = sigma_w / vol
    contribution = w * marginal
    return pd.DataFrame(
        {
            "Peso": w,
            "Contribución marginal": marginal,
            "Contribución al riesgo": contribution,
            "% del riesgo": contribution / vol,
        },
        index=cov.index,
    )


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """(Σ w_i σ_i) / σ_p. Vale 1 si no hay diversificación; crece con ella."""
    w = np.abs(weights.reindex(cov.index).fillna(0.0).values)
    sd = np.sqrt(np.diag(cov.values))
    vol = portfolio_volatility(weights, cov)
    if vol <= 0:
        return float("nan")
    return float((w @ sd) / vol)


def effective_number_of_assets(weights: pd.Series) -> float:
    """1/Σw² — inverso del índice Herfindahl. Cuántos activos "realmente" hay."""
    w = weights.values.astype(float)
    denom = float(np.sum(w**2))
    return 1.0 / denom if denom > 0 else float("nan")


def concentration(weights: pd.Series) -> dict[str, float]:
    w = weights.sort_values(ascending=False)
    positive = w[w > 0]
    return {
        "Mayor posición": float(w.max()) if len(w) else float("nan"),
        "Top 3": float(positive.head(3).sum()),
        "Top 5": float(positive.head(5).sum()),
        "Posiciones activas": float((np.abs(w.values) > 1e-6).sum()),
        "Exposición larga": float(w[w > 0].sum()),
        "Exposición corta": float(w[w < 0].sum()),
        "Apalancamiento bruto": float(np.abs(w.values).sum()),
    }


def summary(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    label: str = "Portafolio",
) -> pd.Series:
    """Ficha de métricas ex-ante de un portafolio."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    conc = concentration(weights)
    return pd.Series(
        {
            "Rendimiento esperado": ret,
            "Volatilidad": vol,
            "Razón de Sharpe": (ret - rf) / vol if vol > 0 else float("nan"),
            "Razón de diversificación": diversification_ratio(weights, cov),
            "Número efectivo de activos": effective_number_of_assets(weights),
            "Mayor posición": conc["Mayor posición"],
            "Top 5": conc["Top 5"],
            "Apalancamiento bruto": conc["Apalancamiento bruto"],
        },
        name=label,
    )


def compare_portfolios(
    portfolios: dict[str, pd.Series],
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
) -> pd.DataFrame:
    """Tabla comparativa de varios portafolios bajo los mismos insumos."""
    return pd.DataFrame({name: summary(w, mu, cov, rf, name) for name, w in portfolios.items()})


def tracking_error(weights: pd.Series, benchmark: pd.Series, cov: pd.DataFrame) -> float:
    """σ(w − w_b): desviación esperada respecto al portafolio de referencia."""
    active = (weights.reindex(cov.index).fillna(0.0) - benchmark.reindex(cov.index).fillna(0.0)).values
    return float(np.sqrt(max(active @ cov.values @ active, 0.0)))


# ==========================================================================
# Gráficas Plotly
# (originalmente blmodel/plots.py)
# ==========================================================================

# --------------------------------------------------------------------------
# Paleta (validada: ΔE CVD ≥ 8 en pares adyacentes, ambos modos)
# --------------------------------------------------------------------------
CATEGORICAL_LIGHT = [
    "#2a78d6",  # 1 azul
    "#eb6834",  # 2 naranja
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 amarillo
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 verde
    "#4a3aa7",  # 7 violeta
    "#e34948",  # 8 rojo
]
CATEGORICAL_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"],
    [0.75, "#2a78d6"], [1.0, "#104281"],
]

THEME = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "neutral": "#f0efec",
        "categorical": CATEGORICAL_LIGHT,
        "positive": "#2a78d6",
        "negative": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "neutral": "#383835",
        "categorical": CATEGORICAL_DARK,
        "positive": "#3987e5",
        "negative": "#e66767",
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(theme: str, title: str, height: int = 420, **kwargs) -> dict:
    t = THEME[theme]
    base = dict(
        title=dict(text=title, font=dict(size=15, color=t["ink"], family=FONT), x=0, xanchor="left"),
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["ink_secondary"], size=12),
        height=height,
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(font=dict(family=FONT, size=12), bordercolor=t["axis"]),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(color=t["ink_secondary"], size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    base.update(kwargs)
    return base


def _axes(fig: go.Figure, theme: str, xgrid: bool = False, ygrid: bool = True,
          xtitle: str = "", ytitle: str = "", tickformat: str | None = None) -> None:
    t = THEME[theme]
    fig.update_xaxes(
        showgrid=xgrid, gridcolor=t["grid"], gridwidth=1, zeroline=False,
        linecolor=t["axis"], linewidth=1, tickfont=dict(color=t["muted"], size=11),
        title=dict(text=xtitle, font=dict(color=t["muted"], size=11)),
    )
    fig.update_yaxes(
        showgrid=ygrid, gridcolor=t["grid"], gridwidth=1, zeroline=False,
        linecolor=t["axis"], linewidth=1, tickfont=dict(color=t["muted"], size=11),
        title=dict(text=ytitle, font=dict(color=t["muted"], size=11)),
        tickformat=tickformat,
    )


# --------------------------------------------------------------------------
# Π vs μ_BL
# --------------------------------------------------------------------------
def plot_returns_comparison(
    pi: pd.Series, mu_bl: pd.Series, theme: str = "light"
) -> go.Figure:
    """Rendimientos de equilibrio contra posteriores, por activo.

    Dos series, mismo eje, mismas unidades: barras agrupadas. La distancia
    entre ambas es, literalmente, el efecto de las views.
    """
    t = THEME[theme]
    order = mu_bl.sort_values(ascending=False).index
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(order), y=pi.reindex(order).values, name="Π (equilibrio)",
        marker=dict(color=t["categorical"][0], line=dict(width=2, color=t["surface"])),
        hovertemplate="<b>%{x}</b><br>Equilibrio: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=list(order), y=mu_bl.reindex(order).values, name="μ_BL (posterior)",
        marker=dict(color=t["categorical"][1], line=dict(width=2, color=t["surface"])),
        hovertemplate="<b>%{x}</b><br>Posterior: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**_layout(theme, "Rendimiento esperado: equilibrio vs posterior"),
                      barmode="group", bargap=0.30, bargroupgap=0.06)
    _axes(fig, theme, ytitle="Rendimiento anual", tickformat=".1%")
    return fig


def plot_view_impact(impact: pd.Series, theme: str = "light") -> go.Figure:
    """μ_BL − Π por activo. Polaridad ⇒ paleta divergente azul/rojo."""
    t = THEME[theme]
    s = impact.sort_values()
    colors = [t["positive"] if v >= 0 else t["negative"] for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=list(s.index), orientation="h",
        marker=dict(color=colors, line=dict(width=2, color=t["surface"])),
        hovertemplate="<b>%{y}</b><br>Ajuste: %{x:+.2%}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_layout(theme, "Ajuste de las views sobre el equilibrio",
                                height=max(280, 30 * len(s) + 110)))
    _axes(fig, theme, xgrid=True, ygrid=False, xtitle="μ_BL − Π")
    fig.update_xaxes(tickformat="+.1%", zeroline=True, zerolinecolor=t["axis"], zerolinewidth=1)
    return fig


# --------------------------------------------------------------------------
# Pesos
# --------------------------------------------------------------------------
def plot_weights_comparison(
    weights: dict[str, pd.Series], theme: str = "light"
) -> go.Figure:
    """Pesos de varios portafolios lado a lado.

    El color sigue al portafolio (la entidad), de modo que agregar o quitar uno
    no repinta a los demás.
    """
    t = THEME[theme]
    names = list(weights.keys())
    reference = weights[names[-1]]
    order = reference.sort_values(ascending=False).index

    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(go.Bar(
            x=list(order), y=weights[name].reindex(order).fillna(0.0).values, name=name,
            marker=dict(color=t["categorical"][i % 8], line=dict(width=2, color=t["surface"])),
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(theme, "Asignación por activo"),
                      barmode="group", bargap=0.30, bargroupgap=0.06)
    _axes(fig, theme, ytitle="Peso", tickformat=".0%")
    fig.update_yaxes(zeroline=True, zerolinecolor=t["axis"], zerolinewidth=1)
    return fig


def plot_allocation(weights: pd.Series, theme: str = "light") -> go.Figure:
    """Asignación del portafolio óptimo, ordenada de mayor a menor.

    Barras horizontales con etiqueta directa: una sola serie, sin leyenda.
    """
    t = THEME[theme]
    s = weights[weights.abs() > 1e-6].sort_values()
    colors = [t["positive"] if v >= 0 else t["negative"] for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=list(s.index), orientation="h",
        marker=dict(color=colors, line=dict(width=2, color=t["surface"])),
        text=[f"{v:.1%}" for v in s.values], textposition="outside",
        textfont=dict(color=t["ink_secondary"], size=11, family=FONT),
        hovertemplate="<b>%{y}</b><br>Peso: %{x:.2%}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_layout(theme, "Portafolio óptimo",
                                height=max(280, 30 * len(s) + 110)))
    _axes(fig, theme, xgrid=True, ygrid=False, xtitle="Peso")
    fig.update_xaxes(tickformat=".0%", zeroline=True, zerolinecolor=t["axis"], zerolinewidth=1)
    return fig


# --------------------------------------------------------------------------
# Frontera eficiente
# --------------------------------------------------------------------------
def plot_efficient_frontier(
    frontier: pd.DataFrame,
    portfolios: dict[str, tuple[float, float]] | None = None,
    theme: str = "light",
) -> go.Figure:
    """Frontera eficiente con los portafolios de interés marcados.

    `portfolios` mapea nombre → (volatilidad, rendimiento).
    """
    t = THEME[theme]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["Volatilidad"], y=frontier["Rendimiento"],
        mode="lines", name="Frontera eficiente",
        line=dict(color=t["muted"], width=2),
        hovertemplate="Vol %{x:.2%}<br>Rend %{y:.2%}<extra></extra>",
    ))
    for i, (name, (vol, ret)) in enumerate((portfolios or {}).items()):
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=name,
            marker=dict(size=13, color=t["categorical"][i % 8],
                        line=dict(width=2, color=t["surface"])),
            text=[name], textposition="top center",
            textfont=dict(color=t["ink_secondary"], size=11, family=FONT),
            hovertemplate=f"<b>{name}</b><br>Vol %{{x:.2%}}<br>Rend %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(theme, "Frontera eficiente", height=460), hovermode="closest")
    _axes(fig, theme, xgrid=True, xtitle="Volatilidad anual", ytitle="Rendimiento esperado",
          tickformat=".1%")
    fig.update_xaxes(tickformat=".1%")
    return fig


# --------------------------------------------------------------------------
# Correlaciones y riesgo
# --------------------------------------------------------------------------
def plot_correlation_heatmap(corr: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Correlaciones: escala divergente con punto medio neutro en cero.

    La correlación tiene polaridad (positiva/negativa), no sólo magnitud: por
    eso azul ↔ gris ↔ rojo, y no un solo tono.
    """
    t = THEME[theme]
    scale = [
        [0.0, "#e34948"], [0.25, "#f0a09f"], [0.5, t["neutral"]],
        [0.75, "#9ec5f4"], [1.0, "#2a78d6"],
    ]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale=scale, zmid=0.0, zmin=-1.0, zmax=1.0,
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="ρ", font=dict(color=t["muted"], size=11)),
                      tickfont=dict(color=t["muted"], size=10), thickness=12, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    n = len(corr)
    fig.update_layout(**_layout(theme, "Matriz de correlaciones", height=max(340, 32 * n + 130)))
    _axes(fig, theme, ygrid=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_risk_contribution(contrib: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Peso vs contribución al riesgo: dónde está el dinero y de dónde viene el riesgo."""
    t = THEME[theme]
    df = contrib[contrib["Peso"].abs() > 1e-6].sort_values("% del riesgo", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(df.index), y=df["Peso"].values, name="Peso del capital",
        marker=dict(color=t["categorical"][0], line=dict(width=2, color=t["surface"])),
        hovertemplate="<b>%{x}</b><br>Peso: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=list(df.index), y=df["% del riesgo"].values, name="Contribución al riesgo",
        marker=dict(color=t["categorical"][1], line=dict(width=2, color=t["surface"])),
        hovertemplate="<b>%{x}</b><br>Riesgo: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**_layout(theme, "Capital vs riesgo"),
                      barmode="group", bargap=0.30, bargroupgap=0.06)
    _axes(fig, theme, ytitle="Proporción del total", tickformat=".0%")
    return fig


# --------------------------------------------------------------------------
# Sensibilidad a τ
# --------------------------------------------------------------------------
def plot_tau_sensitivity(sens: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Pesos implícitos en función de τ.

    Si las líneas son planas, la elección de τ no cambia la recomendación y
    deja de ser un punto de discusión. Si se abren, hay que justificar el valor.
    Más de ocho activos se agrupan en "Otros" en lugar de generar tonos nuevos.
    """
    t = THEME[theme]
    df = sens.copy()
    if df.shape[1] > 8:
        importance = df.abs().max().sort_values(ascending=False)
        keep = list(importance.index[:7])
        others = [c for c in df.columns if c not in keep]
        df = df[keep].assign(Otros=sens[others].sum(axis=1))

    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col].values, mode="lines", name=str(col),
            line=dict(color=t["categorical"][i % 8], width=2),
            hovertemplate=f"<b>{col}</b><br>τ = %{{x:.4f}}<br>Peso %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(theme, "Sensibilidad de los pesos a τ", height=440),
                      hovermode="x unified")
    _axes(fig, theme, xgrid=True, xtitle="τ (incertidumbre del prior)",
          ytitle="Peso implícito", tickformat=".0%")
    return fig


def plot_confidence_sensitivity(sens: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Pesos en función de la confianza declarada en las views (Idzorek)."""
    t = THEME[theme]
    df = sens.copy()
    if df.shape[1] > 8:
        importance = df.abs().max().sort_values(ascending=False)
        keep = list(importance.index[:7])
        others = [c for c in df.columns if c not in keep]
        df = df[keep].assign(Otros=sens[others].sum(axis=1))

    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col].values, mode="lines", name=str(col),
            line=dict(color=t["categorical"][i % 8], width=2),
            hovertemplate=f"<b>{col}</b><br>Confianza %{{x:.0%}}<br>Peso %{{y:.2%}}<extra></extra>",
        ))
    fig.update_layout(**_layout(theme, "Sensibilidad de los pesos a la confianza", height=440),
                      hovermode="x unified")
    _axes(fig, theme, xgrid=True, xtitle="Confianza en las views",
          ytitle="Peso implícito", tickformat=".0%")
    fig.update_xaxes(tickformat=".0%")
    return fig


def plot_view_absorption(diag: pd.DataFrame, theme: str = "light") -> go.Figure:
    """Cuánto del desacuerdo con el equilibrio absorbió el modelo, por view."""
    t = THEME[theme]
    df = diag.copy()
    df["Absorción"] = df["Absorción"].fillna(0.0)
    fig = go.Figure(go.Bar(
        x=df["Absorción"].values, y=df["View"].astype(str).values, orientation="h",
        marker=dict(color=t["categorical"][0], line=dict(width=2, color=t["surface"])),
        text=[f"{v:.0%}" for v in df["Absorción"].values], textposition="outside",
        textfont=dict(color=t["ink_secondary"], size=11, family=FONT),
        hovertemplate="<b>%{y}</b><br>Absorción: %{x:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_layout(theme, "Absorción de cada view",
                                height=max(260, 44 * len(df) + 110)))
    _axes(fig, theme, xgrid=True, ygrid=False,
          xtitle="Proporción del desacuerdo incorporada al posterior")
    fig.update_xaxes(tickformat=".0%", zeroline=True, zerolinecolor=t["axis"], zerolinewidth=1)
    return fig


# ==========================================================================
# Exportación e importación
# (originalmente blmodel/io_utils.py)
# ==========================================================================

# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------
def build_excel(sheets: dict[str, pd.DataFrame], percent_columns: dict[str, list[str]] | None = None) -> bytes:
    """Construye un libro de Excel con una hoja por tabla."""
    buffer = io.BytesIO()
    percent_columns = percent_columns or {}
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        book = writer.book
        pct = book.add_format({"num_format": "0.00%"})
        num = book.add_format({"num_format": "#,##0.0000"})
        header = book.add_format({"bold": True, "bg_color": "#f0efec", "border": 1})

        for name, df in sheets.items():
            safe = name[:31]
            out = df.copy()
            index_label = out.index.name or "Concepto"
            out = out.reset_index().rename(columns={"index": index_label})
            out.to_excel(writer, sheet_name=safe, index=False, startrow=0)
            sheet = writer.sheets[safe]

            for col_idx, col in enumerate(out.columns):
                sheet.write(0, col_idx, str(col), header)
                width = max(12, min(34, int(out[col].astype(str).str.len().max() or 12) + 4))
                if col in percent_columns.get(name, []):
                    sheet.set_column(col_idx, col_idx, width, pct)
                elif pd.api.types.is_numeric_dtype(out[col]):
                    sheet.set_column(col_idx, col_idx, width, num)
                else:
                    sheet.set_column(col_idx, col_idx, width)
            sheet.freeze_panes(1, 1)
    buffer.seek(0)
    return buffer.getvalue()


def build_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
def export_config(cfg: ModelConfig, extra: dict | None = None) -> bytes:
    payload = cfg.to_dict()
    payload["_exportado"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        payload["_contexto"] = extra
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def import_config(raw: bytes | str) -> ModelConfig:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return ModelConfig.from_json(text)


# --------------------------------------------------------------------------
# Views en CSV
# --------------------------------------------------------------------------
VIEW_CSV_COLUMNS = ["tipo", "activos", "q", "confianza", "etiqueta", "activa"]


def views_to_csv(views: list[ViewSpec]) -> bytes:
    """Serializa las views a CSV.

    La columna `activos` usa la sintaxis "TICKER:peso" separada por punto y
    coma, por ejemplo: "NVDA:1;KO:-1".
    """
    rows = []
    for v in views:
        rows.append({
            "tipo": v.kind.value,
            "activos": ";".join(f"{a}:{w:g}" for a, w in v.assets.items()),
            "q": v.q,
            "confianza": v.confidence,
            "etiqueta": v.label,
            "activa": int(v.enabled),
        })
    df = pd.DataFrame(rows, columns=VIEW_CSV_COLUMNS)
    return df.to_csv(index=False).encode("utf-8-sig")


def views_from_csv(raw: bytes | str) -> tuple[list[ViewSpec], list[str]]:
    """Lee views desde CSV. Devuelve las válidas y los errores encontrados."""
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    errors: list[str] = []
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        return [], [f"No se pudo leer el CSV: {exc}"]

    missing = [c for c in ("activos", "q") if c not in df.columns]
    if missing:
        return [], [f"Faltan columnas obligatorias: {', '.join(missing)}."]

    views: list[ViewSpec] = []
    for i, row in df.iterrows():
        try:
            assets: dict[str, float] = {}
            for part in str(row["activos"]).split(";"):
                part = part.strip()
                if not part:
                    continue
                ticker, _, weight = part.partition(":")
                assets[ticker.strip().upper()] = float(weight) if weight else 1.0
            if not assets:
                errors.append(f"Fila {i + 2}: sin activos.")
                continue
            kind_raw = str(row.get("tipo", "")).strip().lower()
            kind = ViewKind.RELATIVE if kind_raw.startswith("rel") else ViewKind.ABSOLUTE
            views.append(ViewSpec(
                kind=kind,
                assets=assets,
                q=float(row["q"]),
                confidence=float(row.get("confianza", 0.5)),
                label=str(row.get("etiqueta", "") or ""),
                enabled=bool(int(row.get("activa", 1))),
            ))
        except Exception as exc:
            errors.append(f"Fila {i + 2}: {exc}")
    return views, errors


def views_template_csv() -> bytes:
    """Plantilla de ejemplo para que el usuario sepa el formato esperado."""
    sample = pd.DataFrame(
        [
            {"tipo": "absoluta", "activos": "AAPL:1", "q": 0.12, "confianza": 0.60,
             "etiqueta": "AAPL rinde 12% anual", "activa": 1},
            {"tipo": "relativa", "activos": "MSFT:1;KO:-1", "q": 0.05, "confianza": 0.40,
             "etiqueta": "MSFT supera a KO por 5%", "activa": 1},
        ],
        columns=VIEW_CSV_COLUMNS,
    )
    return sample.to_csv(index=False).encode("utf-8-sig")


# ==========================================================================
# Orquestación: de la configuración al portafolio óptimo
# (originalmente blmodel/pipeline.py)
# ==========================================================================

@dataclass
class ModelRun:
    """Resultado completo de una corrida."""

    config: ModelConfig
    prices: pd.DataFrame
    returns: pd.DataFrame
    validation: ValidationReport
    cov: pd.DataFrame
    cov_diag: CovDiagnostics
    prior: PriorResult
    views: ViewMatrices
    bl: BLResult
    portfolio: PortfolioResult
    benchmark_returns: pd.Series | None = None
    frontier: pd.DataFrame | None = None
    comparison: pd.DataFrame | None = None
    view_diag: pd.DataFrame | None = None
    risk_contrib: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def assets(self) -> list[str]:
        return list(self.cov.index)

    def summary_row(self) -> pd.Series:
        return metrics.summary(
            self.portfolio.weights, self.bl.mu_bl, self.bl.cov_bl,
            self.prior.risk_free, "Black-Litterman",
        )


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
@dataclass
class MarketData:
    """Insumos crudos, separados del modelo para poder cachearlos."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    validation: ValidationReport
    benchmark_returns: pd.Series | None
    market_caps: pd.Series | None
    risk_free: float
    rf_source: str
    missing_caps: list[str] = field(default_factory=list)


def load_market_data(cfg: ModelConfig, fetch_caps: bool = True) -> MarketData:
    """Descarga y valida todo lo que el modelo necesita de Yahoo Finance."""
    start, end = resolve_window(cfg.data)
    tickers = list(cfg.data.tickers)
    if not tickers:
        raise DataError("Especifique al menos dos tickers.")

    symbols = list(dict.fromkeys(tickers + ([cfg.data.benchmark] if cfg.data.benchmark else [])))
    raw = download_prices(symbols, start, end, cfg.data.frequency)

    bench_prices = None
    if cfg.data.benchmark and cfg.data.benchmark.upper() in raw.columns:
        bench_prices = raw[cfg.data.benchmark.upper()].dropna()

    prices, report = validate_prices(raw, tickers, cfg.data)
    if not report.ok:
        raise DataError(
            "Datos insuficientes para optimizar. " + " ".join(report.warnings)
        )

    returns = compute_returns(prices, cfg.data.return_type)

    bench_returns = None
    if bench_prices is not None and len(bench_prices) > 2:
        bench_returns = compute_returns(
            bench_prices.to_frame("bench"), cfg.data.return_type
        )["bench"]
        bench_returns = bench_returns.reindex(returns.index).dropna()

    caps, missing = (pd.Series(dtype=float), list(report.accepted))
    if fetch_caps:
        caps, missing = fetch_market_caps(report.accepted)

    if cfg.prior.rf_auto and cfg.prior.rf_manual is None:
        rf, rf_source = fetch_risk_free_rate(cfg.prior.rf_ticker)
    else:
        rf = float(cfg.prior.rf_manual if cfg.prior.rf_manual is not None else 0.04)
        rf_source = f"Tasa libre de riesgo fijada manualmente en {rf:.2%}."

    return MarketData(
        prices=prices, returns=returns, validation=report,
        benchmark_returns=bench_returns, market_caps=caps if len(caps) else None,
        risk_free=rf, rf_source=rf_source, missing_caps=missing,
    )


# --------------------------------------------------------------------------
# Corrida del modelo
# --------------------------------------------------------------------------
def run_model(
    cfg: ModelConfig,
    data: MarketData,
    include_estimation_risk: bool = True,
    build_frontier: bool = True,
) -> ModelRun:
    """Encadena covarianza → prior → views → posterior → optimización."""
    ppy = cfg.data.frequency.periods_per_year
    notes: list[str] = []

    cov, cov_diag = estimate_covariance(data.returns, cfg.covariance, ppy)
    notes.extend(cov_diag.notes())

    prior = build_prior(
        cov=cov,
        cfg=cfg.prior,
        benchmark_returns=data.benchmark_returns,
        periods_per_year=ppy,
        n_observations=len(data.returns),
        risk_free=data.risk_free,
        rf_source=data.rf_source,
        market_caps=data.market_caps,
    )
    notes.extend(prior.fallbacks)

    views = build_view_matrices(
        cfg=cfg.views, assets=list(cov.index), pi=prior.pi, cov=cov,
        tau=prior.tau, delta=prior.delta, w_market=prior.weights,
    )
    notes.extend(views.warnings)

    bl = run_black_litterman(
        pi=prior.pi, cov=cov, tau=prior.tau, delta=prior.delta,
        views=views, include_estimation_risk=include_estimation_risk,
    )

    portfolio = optimize(
        mu=bl.mu_bl, cov=bl.cov_bl, cfg=cfg.optimizer,
        risk_free=prior.risk_free, delta=prior.delta,
    )

    frontier = None
    if build_frontier:
        try:
            frontier = efficient_frontier(bl.mu_bl, bl.cov_bl, cfg.optimizer)
        except OptimizationError as exc:
            notes.append(f"No se pudo trazar la frontera eficiente: {exc}")

    comparison = _build_comparison(cfg, bl, prior, portfolio, notes)
    vdiag = view_diagnostics(bl, views)
    rcontrib = metrics.risk_contributions(portfolio.weights, bl.cov_bl)

    return ModelRun(
        config=cfg, prices=data.prices, returns=data.returns,
        validation=data.validation, cov=cov, cov_diag=cov_diag, prior=prior,
        views=views, bl=bl, portfolio=portfolio,
        benchmark_returns=data.benchmark_returns, frontier=frontier,
        comparison=comparison, view_diag=vdiag, risk_contrib=rcontrib, notes=notes,
    )


def _build_comparison(
    cfg: ModelConfig,
    bl: BLResult,
    prior: PriorResult,
    portfolio: PortfolioResult,
    notes: list[str],
) -> pd.DataFrame:
    """Compara el portafolio BL contra referencias bajo los mismos insumos."""
    portfolios: dict[str, pd.Series] = {
        "Mercado (equilibrio)": prior.weights,
        "Black-Litterman": portfolio.weights,
    }
    try:
        gmv, _ = min_variance(bl.cov_bl, cfg.optimizer)
        portfolios["Mínima varianza"] = gmv
    except OptimizationError as exc:
        notes.append(f"No se pudo calcular el portafolio de mínima varianza: {exc}")

    n = len(bl.mu_bl)
    portfolios["Equiponderado"] = pd.Series(np.full(n, 1.0 / n), index=bl.mu_bl.index)

    return metrics.compare_portfolios(portfolios, bl.mu_bl, bl.cov_bl, prior.risk_free)


# --------------------------------------------------------------------------
# Análisis de sensibilidad
# --------------------------------------------------------------------------
def tau_sensitivity_analysis(
    run: ModelRun,
    tau_min: float = 0.005,
    tau_max: float = 1.0,
    n_points: int = 24,
    constrained: bool = True,
) -> pd.DataFrame:
    """Pesos óptimos en función de τ, bajo las restricciones configuradas."""
    taus = np.geomspace(tau_min, tau_max, n_points)

    def weight_fn(mu: pd.Series, tau: float) -> pd.Series:
        if not constrained:
            return unconstrained_weights(mu, run.cov, run.prior.delta)
        try:
            return optimize(
                mu, run.cov, run.config.optimizer,
                risk_free=run.prior.risk_free, delta=run.prior.delta,
            ).weights
        except OptimizationError:
            return pd.Series(np.nan, index=run.cov.index)

    return tau_sensitivity(
        pi=run.prior.pi, cov=run.cov, delta=run.prior.delta,
        views=run.views, taus=taus, weight_fn=weight_fn,
    )


def confidence_sensitivity_analysis(
    run: ModelRun,
    n_points: int = 11,
    constrained: bool = True,
) -> pd.DataFrame:
    """Pesos óptimos al mover *todas* las confianzas de 0% a 100% en bloque.

    Sólo tiene sentido con Ω por Idzorek, que es donde la confianza es un
    parámetro explícito.
    """
    if run.views.n_views == 0 or run.config.views.method is not OmegaMethod.IDZOREK:
        return pd.DataFrame()


    cov_np = np.asarray(run.cov.values, dtype=float)
    pi_np = np.asarray(run.prior.pi.values, dtype=float)
    w_np = np.asarray(run.prior.weights.values, dtype=float)
    levels = np.linspace(0.0, 1.0, n_points)

    rows = []
    for c in levels:
        omega = omega_idzorek(
            run.views.P, run.views.Q, [float(c)] * run.views.n_views,
            pi_np, run.prior.tau, cov_np, run.prior.delta, w_np,
        )
        mu = posterior_returns(pi_np, cov_np, run.prior.tau, run.views.P, run.views.Q, omega)
        mu_s = pd.Series(mu, index=run.cov.index)
        if constrained:
            try:
                w = optimize(
                    mu_s, run.cov, run.config.optimizer,
                    risk_free=run.prior.risk_free, delta=run.prior.delta,
                ).weights
            except OptimizationError:
                w = pd.Series(np.nan, index=run.cov.index)
        else:
            w = unconstrained_weights(mu_s, run.cov, run.prior.delta)
        rows.append(w.rename(float(c)))
    return pd.DataFrame(rows)


def prior_comparison(run: ModelRun, data: MarketData) -> pd.DataFrame:
    """Π resultante bajo cada fuente de pesos de mercado.

    Responde la pregunta que el comité siempre hace: "¿y si los pesos de
    referencia fueran otros?".
    """

    assets = run.assets
    out: dict[str, pd.Series] = {}
    for source in WeightSource:
        cfg = PriorConfig(**{**run.config.prior.__dict__, "weight_source": source})
        try:
            w, label, _ = resolve_market_weights(assets, cfg, data.market_caps, None)
            out[label] = implied_equilibrium_returns(run.prior.delta, run.cov, w)
        except Exception:
            continue
    return pd.DataFrame(out)


def covariance_comparison(run: ModelRun) -> pd.DataFrame:
    """Volatilidades anualizadas bajo cada estimador de Σ, para contraste."""

    ppy = run.config.data.frequency.periods_per_year
    out: dict[str, pd.Series] = {}
    for method in CovMethod:
        cfg = CovConfig(**{**run.config.covariance.__dict__, "method": method})
        try:
            cov, _ = estimate_covariance(run.returns, cfg, ppy)
            out[method.value] = annualized_volatilities(cov)
        except Exception:
            continue
    return pd.DataFrame(out)

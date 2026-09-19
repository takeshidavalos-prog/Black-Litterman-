# Modelo Black-Litterman

Aplicación de optimización de portafolios que combina el equilibrio de mercado
con las views del inversionista, sobre datos de Yahoo Finance. Paquete de Python
modular con interfaz en Streamlit.

---

## Qué resuelve

La optimización media-varianza clásica tiene un defecto práctico conocido: es
extremadamente sensible a los rendimientos esperados que se le alimentan. Si se
usan medias históricas, el optimizador devuelve portafolios esquinados —todo el
capital en el activo que casualmente subió más— y esos pesos cambian
drásticamente ante variaciones mínimas del insumo.

Black-Litterman invierte el planteamiento. En lugar de pedir rendimientos
esperados, parte de los que **ya están implícitos en los precios**, y sólo se
aparta de ellos en la medida en que el inversionista declara una opinión y una
confianza. Sin opiniones, el resultado es el portafolio de mercado. Con ellas, el
desvío es proporcional a la convicción declarada y respeta la estructura de
correlaciones.

---

## Formulación

### 1. Prior de equilibrio

Si el portafolio de mercado es el óptimo del inversionista representativo, los
rendimientos esperados se obtienen por optimización inversa:

$$\Pi = \delta \cdot \Sigma \cdot w_{mkt}$$

| Símbolo | Significado | Cómo se obtiene en la app |
|---|---|---|
| $\Pi$ | Rendimientos de equilibrio | Calculado |
| $\delta$ | Aversión al riesgo | $(E[R_b] - r_f)/\sigma_b^2$ del benchmark, o manual |
| $\Sigma$ | Covarianza anualizada | Cinco estimadores seleccionables |
| $w_{mkt}$ | Pesos de mercado | Capitalización de Yahoo, benchmark, propios o equiponderado |

La identidad $(\delta\Sigma)^{-1}\Pi = w_{mkt}$ se verifica como prueba de
regresión: es lo que hace del prior un prior *de equilibrio*.

### 2. Views

Cada view se expresa como una fila de $P$ y una entrada de $Q$:

- **Absoluta** — "NVDA rendirá 15%": $P_k = [0,\ldots,1,\ldots,0]$, $Q_k = 0.15$
- **Relativa** — "NVDA superará a KO por 5%": $P_k = [0,\ldots,1,\ldots,-1,\ldots,0]$, $Q_k = 0.05$

En una view relativa los pesos deben sumar cero; en una absoluta, uno. La
aplicación lo valida y descarta las views mal formadas, avisando cuáles.

### 3. Matriz de confianza Ω

$\Omega$ es la varianza del error de cada view. Dos métodos, ambos disponibles:

**Idzorek (2005)** — el usuario declara confianza de 0% a 100% y el método
resuelve numéricamente el $\omega$ que produce exactamente ese desplazamiento:

| Confianza | Portafolio resultante |
|---|---|
| 0% | El de mercado: la view se ignora |
| 50% | A la mitad exacta del camino |
| 100% | El de confianza total: $P\mu_{BL} = Q$ |

**He-Litterman** — $\Omega = \mathrm{diag}(P \cdot \tau\Sigma \cdot P^T)$. Sin
parámetro subjetivo: cada view hereda la incertidumbre que el prior ya tiene en
esa dirección. Todas pesan igual por construcción.

### 4. Posterior

$$\mu_{BL} = \Pi + \tau\Sigma P^T (P\tau\Sigma P^T + \Omega)^{-1}(Q - P\Pi)$$

Algebraicamente equivalente a la forma de precisiones
$\left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1}\left[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q\right]$,
pero invierte una matriz $k \times k$ (número de views) en lugar de una
$n \times n$, y no exige que $\Omega$ sea invertible término a término. La
equivalencia entre ambas formas está cubierta por una prueba.

La covarianza posterior incorpora el riesgo de estimación:

$$\Sigma_{BL} = \Sigma + M, \qquad M = \tau\Sigma - \tau\Sigma P^T(P\tau\Sigma P^T + \Omega)^{-1}P\tau\Sigma$$

### 5. Optimización

Todos los objetivos se resuelven como problemas convexos con `cvxpy`. La máxima
razón de Sharpe, que no es convexa en $w$, se resuelve por la transformación
homogénea de Cornuejols-Tütüncü ($y = w/\kappa$), lo que permite maximizar
Sharpe **con** cotas por activo y por grupo — algo que la fórmula analítica no
admite.

---

## Instalación

```bash
git clone https://github.com/<usuario>/black-litterman-streamlit.git
cd black-litterman-streamlit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`.

### Despliegue en Streamlit Community Cloud

1. Suba el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), conecte el repositorio.
3. Archivo principal: `app.py`. No requiere variables de entorno ni credenciales:
   Yahoo Finance se consulta sin llave.

---

## Uso

1. **Universo** — escriba los tickers en el panel izquierdo. Funciona cualquier
   símbolo de Yahoo Finance: acciones (`AAPL`), ETFs (`SPY`), emisoras mexicanas
   (`WALMEX.MX`), índices (`^GSPC`), cripto (`BTC-USD`).
2. **Benchmark** — se usa para estimar $\delta$. Por defecto `^GSPC`; para
   México, `^MXX`.
3. **Views** — agregue renglones en la tabla central. El campo *Activos* usa la
   sintaxis `TICKER:peso` separada por punto y coma:
   - Absoluta: `NVDA:1` con Q = 15
   - Relativa: `NVDA:1;KO:-1` con Q = 5
4. **Ejecutar modelo.**

Sin views, el resultado es el portafolio de equilibrio: es el comportamiento
correcto, no un error.

---

## Configuración disponible

| Bloque | Opciones |
|---|---|
| **Datos** | Frecuencia diaria/semanal/mensual · ventana en años o fechas exactas · rendimientos simples o logarítmicos · umbrales de validación |
| **Covarianza** | Muestral · Ledoit-Wolf · OAS · EWMA (λ ajustable) · semicovarianza · verificación y reparación PSD |
| **Prior** | Pesos por capitalización, benchmark, propios o equiponderado · δ implícito o manual · τ manual o 1/T · tasa libre de riesgo de ^IRX/^FVX/^TNX/^TYX o manual |
| **Views** | Absolutas y relativas · Ω por Idzorek, He-Litterman o manual · import/export CSV |
| **Optimizador** | Máx Sharpe · mín varianza · máx utilidad · vol objetivo · rendimiento objetivo · long-only o cortos · cotas por activo y por grupo · apalancamiento bruto |

---

## Qué entrega la interfaz

- **Resumen** — pesos óptimos, métricas ex-ante y comparativa contra el
  portafolio de mercado, el de mínima varianza y el equiponderado, todos
  evaluados con los mismos insumos.
- **Datos** — reporte de validación ticker por ticker, diagnóstico de Σ (número
  de condición, valor propio mínimo, intensidad de shrinkage) y contraste de
  volatilidades entre los cinco estimadores.
- **Prior** — Π resultante bajo cada fuente de pesos, para responder la pregunta
  de qué pasaría con otra referencia.
- **Views** — tabla de absorción: qué decía usted, qué decía el mercado, dónde
  quedó el posterior y qué proporción del desacuerdo se incorporó.
- **Posterior** — Π contra μ_BL por activo, y el ajuste que imprimió cada view.
- **Optimización** — asignación, frontera eficiente, descomposición del riesgo y
  el costo de las restricciones contra la solución cerrada.
- **Sensibilidad** — cómo se mueven los pesos al variar τ y al variar la
  confianza declarada.
- **Exportación** — libro de Excel con nueve hojas, CSV de pesos y JSON de
  configuración que reconstruye la corrida completa.

---

## Estructura

```
├── app.py                      Interfaz Streamlit (delgada: sólo presentación)
├── blmodel/
│   ├── config.py               Configuración serializable de una corrida
│   ├── data.py                 Descarga y validación de Yahoo Finance
│   ├── covariance.py           Estimadores de Σ y reparación PSD
│   ├── prior.py                Optimización inversa: δ, w, τ, Π
│   ├── views.py                P, Q y Ω (Idzorek / He-Litterman)
│   ├── blacklitterman.py       Posterior bayesiano
│   ├── optimizer.py            Optimización convexa con restricciones
│   ├── metrics.py              Métricas ex-ante y descomposición de Euler
│   ├── plots.py                Gráficas Plotly
│   ├── io_utils.py             Exportación e importación
│   └── pipeline.py             Orquestación de extremo a extremo
├── tests/                      179 pruebas
└── .github/workflows/ci.yml    Lint y pruebas en Python 3.10–3.12
```

El paquete no depende de Streamlit: `blmodel` se puede usar desde un script, un
notebook o un servicio.

```python
from blmodel import ModelConfig, ViewSpec, ViewKind, load_market_data, run_model

cfg = ModelConfig()
cfg.data.tickers = ["SPY", "EFA", "EEM", "AGG", "GLD"]
cfg.views.views = [
    ViewSpec(kind=ViewKind.RELATIVE, assets={"GLD": 1.0, "AGG": -1.0},
             q=0.04, confidence=0.60, label="Oro supera a bonos por 4%"),
]

run = run_model(cfg, load_market_data(cfg))
print(run.portfolio.weights)
print(run.bl.comparison_frame())
```

---

## Validación

```bash
pip install -r requirements-dev.txt
pytest --cov=blmodel
ruff check blmodel tests app.py
```

179 pruebas, 88% de cobertura sobre el paquete. La suite no toca la red: la
descarga está aislada en `blmodel/data.py` y se prueba contra matrices de
precios construidas a mano.

El ancla de la suite es la **reproducción del ejemplo de He y Litterman (1999)**,
siete mercados accionarios desarrollados:

| | AUS | CAN | FRA | GER | JAP | UKG | USA |
|---|---|---|---|---|---|---|---|
| **Π publicado** | 3.9% | 6.9% | 8.4% | 9.0% | 4.3% | 6.8% | 7.6% |
| **Π calculado** | 3.9% | 6.9% | 8.4% | 9.0% | 4.3% | 6.8% | 7.6% |
| **μ_BL publicado** | 4.3% | 7.6% | 9.3% | 11.0% | 4.5% | 7.0% | 8.1% |
| **μ_BL calculado** | 4.3% | 7.6% | 9.3% | 11.0% | 4.5% | 7.0% | 8.1% |
| **w publicado** | 1.5% | 2.1% | −4.0% | 35.4% | 11.0% | −9.5% | 58.6% |
| **w calculado** | 1.5% | 2.1% | −4.0% | 35.4% | 11.0% | −9.5% | 58.6% |

Que los pesos cuadren con $\Sigma + M$ y no con $\Sigma$ a secas confirma que el
paper original incorpora el riesgo de estimación, igual que esta implementación.

Además se verifica que:

- Sin views, el posterior es idéntico al prior y el óptimo es el portafolio de mercado.
- Con confianza 100%, la view se cumple exactamente: $P\mu_{BL} = Q$.
- Con confianza 50%, el peso se mueve exactamente la mitad — probado en seis
  niveles de confianza, activo por activo.
- Las contribuciones al riesgo suman exactamente la volatilidad del portafolio.
- Cada restricción (cotas, grupos, apalancamiento) se respeta bajo los cinco objetivos.
- La configuración exportada reproduce el mismo portafolio, bit a bit.

---

## Decisiones de diseño

**Se reporta, no se corrige en silencio.** Los valores atípicos se detectan y se
informan, pero no se eliminan: borrar un movimiento extremo real sesga Σ a la
baja. Cuando Yahoo no reporta capitalización para parte del universo, el prior
cae a equiponderado **completo** en lugar de mezclar dos criterios de peso, y lo
asienta en pantalla. Cada respaldo activado aparece como advertencia.

**Σ + M, no Σ.** La covarianza posterior incorpora el riesgo de estimación. Es
lo que hace el paper original y la diferencia sí mueve los pesos. La opción de
usar Σ a secas existe en el código (`include_estimation_risk=False`) para poder
reproducir implementaciones que la omiten.

**δ acotado.** Un δ estimado en una ventana bajista sale negativo, lo que
invertiría el signo de todo el prior. Se acota al rango [0.5, 10] y se avisa en
lugar de propagar el disparate.

**Paleta validada.** Los colores de las gráficas pasan las pruebas de separación
para daltonismo (ΔE ≥ 8 en pares adyacentes, ambos modos). Los tonos se asignan
en orden fijo, nunca cíclicamente: con más de ocho series el exceso se agrupa en
"Otros" en vez de generar colores nuevos.

---

## Limitaciones

- **No incluye backtesting.** La aplicación entrega el portafolio óptimo y sus
  métricas ex-ante; no evalúa cómo se habría comportado históricamente.
- **Sin costos de transacción ni impuestos.** Los pesos son objetivos, no un
  plan de ejecución.
- **Divisa única.** Los activos se tratan en la moneda en que Yahoo los reporta;
  no hay conversión ni tratamiento explícito del riesgo cambiario.
- **El universo restringido no es el mercado.** Al limitar el análisis a unos
  cuantos activos, $w_{mkt}$ son los pesos del subuniverso, no del mercado
  global. Es la aproximación estándar, pero conviene declararla.
- **Yahoo Finance es una fuente no contractual.** Puede cambiar su API, omitir
  campos o devolver datos incompletos. Por eso toda descarga pasa por validación
  antes de entrar al modelo.

---

## Referencias

- Black, F. y Litterman, R. (1992). *Global Portfolio Optimization*. Financial
  Analysts Journal, 48(5), 28–43.
- He, G. y Litterman, R. (1999). *The Intuition Behind Black-Litterman Model
  Portfolios*. Goldman Sachs Investment Management Research.
- Idzorek, T. (2005). *A Step-by-Step Guide to the Black-Litterman Model:
  Incorporating User-Specified Confidence Levels*. Ibbotson Associates.
- Ledoit, O. y Wolf, M. (2004). *A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis.
- Meucci, A. (2010). *The Black-Litterman Approach: Original Model and
  Extensions*. Encyclopedia of Quantitative Finance.
- Cornuejols, G. y Tütüncü, R. (2007). *Optimization Methods in Finance*.
  Cambridge University Press.

---

## Licencia

MIT. Ver [LICENSE](LICENSE).

---

## Aviso

Este software es una herramienta de análisis cuantitativo con fines educativos y
profesionales. No constituye asesoría de inversión ni una recomendación de
compra o venta. Los rendimientos esperados son estimaciones sujetas a error de
modelo y de estimación; el desempeño histórico no garantiza resultados futuros.
Las decisiones de inversión son responsabilidad de quien las toma.

# NeumonAI — Multi-Model AI System for Pneumonia Detection

Sistema multi-paradigma de Inteligencia Artificial para la **detección automatizada de neumonía** a partir de radiografías de tórax (*Chest X-Ray*), evaluando y contrastando cuatro enfoques complementarios sobre el conjunto de datos **Chest X-Ray Images (Pneumonia)** de Kaggle (`NORMAL` vs. `PNEUMONIA`):

1. **Deep Learning End-to-End (CNN)** con diseño experimental factorial $2^4$.
2. **Optimización Metaheurística de CNN** mediante Colonia de Hormigas (**ACO**), Algoritmos Genéticos (**GA**) y Enjambre de Partículas (**PSO**).
3. **Razonamiento Probabilístico Causal (Red Bayesiana)** sobre 18 signos radiológicos extraídos con **TorchXRayVision**.
4. **Ensambles Basados en Árboles (Random Forest y Gradient Boosting)** sobre características clínicas continuas de **TorchXRayVision**.

---

## Resumen Comparativo de Resultados

La siguiente tabla consolida el desempeño en el conjunto de prueba (*Test*) del mejor modelo obtenido en cada enfoque:

| Enfoque | Mejor Modelo / Configuración | Accuracy | Precision | Recall (Sensibilidad) | F1-Score | ROC-AUC | 
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. CNN (Diseño Factorial)** | `aug=No \| lr=0.001 \| do=0.3 \| bl=2` *(Rank 1 Val)* | 76.8% | 73.2% | **99.3%** | 0.843 | 0.908 | 
| **1. CNN (Mejor Balance Test)** | `aug=No \| lr=0.001 \| do=0.3 \| bl=3` *(Rank 2 Val)* | 80.3% | 76.8% | 98.2% | 0.862 | 0.925 | 
| **2. CNN + Metaheurísticas** | `E2_Solo_GA` *(EvoCNN)* | 83.98% | 83.52% | 92.85% | 0.8789 | 0.9129 | 
| **3. TorchXRayVision + Red Bayesiana** | `Exp_33` *(5 Estados, Hill Climbing + BIC, BDeu=50, Ablación)* | 91.19% | 92.80% | 95.13% | 0.9395 | 0.9515 | 
| **4. TorchXRayVision + Ensambles** | `GB_25` *(Gradient Boosting)* | **94.13%** | **96.04%** | 95.78% | **0.9591** | **0.9760** |

---

## Limpieza de Datos 


- **Auditoría de integridad**: Verificación de archivos corruptos (0 encontrados), modos de color (`L`: 5,573; `RGB`: 283) y eliminación de **32 duplicados exactos** mediante hash MD5, reduciendo el conjunto a **5,824 imágenes** (o **5,277** imágenes curadas en el manifiesto utilizado por los pipelines de TorchXRayVision tras excluir 579 registros).
- **Partición libre de fuga por paciente**: Dado que el split `val` original solo contenía 16 imágenes, se unificaron `train + val` y se re-dividieron (80/20) usando **`StratifiedGroupKFold`** agrupado por identificador de paciente (`person<id>` / `IM-<id>`), garantizando conjuntos disjuntos de pacientes entre entrenamiento (`4,168`), validación (`1,038`) y prueba (`618`).
- **Prevención de fuga semántica (Oracle Leakage)**: En los modelos tabulares con TorchXRayVision se excluyó explícitamente la variable de salida preentrenada `Pneumonia` junto con los metadatos (`path`, `partition`, `group`), obligando a los modelos a diagnosticar únicamente a partir de signos clínicos secundarios y comorbilidades radiológicas.

---

## 1. Clasificación End-to-End con CNN ([`CNN.ipynb`](01_CNN_Model/CNN.ipynb))

### Técnicas Aplicadas
- **Preprocesamiento**: Conversión a escala de grises (`150×150×1`), reescalado `[0, 1]` integrado en el grafo del modelo y **pesos de clase balanceados** (`{0: 1.95, 1: 0.67}`) para mitigar el desbalanceo (74.4% neumonía en `train`).
- **Arquitectura Base**: Bloques secuenciales `Conv2D(32·2^i, 3×3, ReLU) + MaxPooling2D(2)` → `Flatten` → `Dense(128, ReLU)` → `Dropout` → `Dense(1, Sigmoid)`.
- **Diseño Experimental Factorial Completo ($2^4 = 16$ configuraciones × 3 semillas = 48 corridas)**:
  - *Aumento de datos*: `No` vs. `Sí` (`RandomRotation(0.05)`, `RandomZoom(0.1)`, `RandomTranslation(0.05)`, `RandomContrast(0.1)`).
  - *Learning rate*: `1e-3` vs. `1e-4` (optimizador Adam, 3 épocas, batch 32, umbral 0.5).
  - *Dropout*: `0.3` vs. `0.5`.
  - *Profundidad*: `2` vs. `3` bloques convolucionales.

### Métricas y Hallazgos
- **Top 1 (por F1 en validación)**: `aug=No | lr=0.001 | do=0.3 | bl=2` obtuvo `val_f1 = 0.977 ± 0.002` y `val_recall = 0.977 ± 0.005`. En test alcanzó `Recall = 0.993 ± 0.004`, `F1 = 0.843 ± 0.002`, `Precision = 0.732 ± 0.002`, `Accuracy = 0.768 ± 0.002`, `AUC = 0.908 ± 0.017`.
- **Comparativa controlada (`SEED=42`) vs. Baseline (`do=0.5, bl=2`)**: El ajuste de Dropout a `0.3` incrementó el **Recall clínico en +0.018** (de `0.979` a `0.997`) y el **AUC** de `0.904` a `0.916`.
- **Efectos principales**: Bajo entrenamiento corto (3 épocas), el aumento de datos penalizó la convergencia (`val_f1` 0.895 vs. 0.966 sin aumento), mientras que `lr=0.001` y `dropout=0.3` maximizaron el Recall y la estabilidad.

---

## 2. Optimización Metaheurística de CNN: ACO, GA y PSO ([`CNN+AG.ipynb`](02_CNN_Genetic_Algorithm/CNN+AG.ipynb))

### Técnicas Aplicadas
Sobre el mismo pipeline de datos curados, se implementaron tres metaheurísticas fundamentadas en literatura científica para optimizar topología, hiperparámetros y pesos:

1. **ACO (*DeepSwarm* — Byla & Pang, 2019)**: Búsqueda de topología sobre un grafo dirigido acíclico (DAG) para seleccionar filtros, tamaño de kernel ($3\times 3$ vs. $5\times 5$) y *pooling* (`max` vs. `avg`), aplicando regla pseudoaleatoria ($q_0 = 0.50$), heurística local $\eta$ y desgaste local de feromona ($\xi = 0.10$).
   - *Topología óptima*: 2 bloques convolucionales (`16 filtros 5×5 + MaxPool` → `64 filtros 3×3 + MaxPool`), fitness de validación `0.9400`.
2. **GA (*EvoCNN* — Sun et al., 2020; Nwohiri et al., 2026)**: Evolución de `lr`, `dropout` acotado en $[0.15, 0.40]$, `dense_units` y parámetros $(\mu, \sigma)$ del inicializador gaussiano de pesos, con selección por torneo con holgura ($\epsilon = 0.01$).
   - *Hiperparámetros óptimos*: `lr=0.001`, `dropout≈0.2995`, `dense_units=128`, `μ=0.0`, `σ=0.05`, fitness de validación `0.9426`.
3. **PSO (*ZooCNN* — Ganesan et al., 2025)**: Optimización de orden cero en el subespacio de la capa densa de salida maximizando un **fitness clínico compuesto** ($0.50\cdot\text{F1} + 0.25\cdot\text{Sensibilidad} + 0.25\cdot\text{Especificidad}$), contrastando:
   - `PSO Naive`: inicialización aleatoria en $[-1.5, 1.5]$ sin *warm-start* e inercia fija $w = 1.0$.
   - `PSO ZooCNN`: *warm-start* desde pesos de Adam, inercia linealmente decreciente ($w: 0.9 \to 0.4$), $c_1 = c_2 = 1.5$ y acotamiento de velocidad ($V_{\max} = 0.10$).

### Métricas por Tratamiento en Test ($N_{\text{reps}} = 3$)

| Tratamiento | Accuracy | Precision | Sensibilidad | Especificidad | F1-Score (Media ± DE) | ROC-AUC (Media ± DE) | Tiempo (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E0_Baseline_CNN` | 80.31% | 78.60% | 94.23% | 57.00% | `0.8571 ± 0.0194` | `0.8998 ± 0.0317` | 14.39 |
| `E1_Solo_ACO` | 79.29% | 77.79% | 94.23% | 54.26% | `0.8511 ± 0.0134` | **`0.9132 ± 0.0047`** | **13.74** |
| **`E2_Solo_GA`** | **83.98%** | **83.52%** | 92.85% | **69.12%** | **`0.8789 ± 0.0094`** | `0.9129 ± 0.0054` | 14.78 |
| `E3_Solo_PSO_Naive` | 76.21% | 73.59% | **96.90%** | 41.56% | `0.8363 ± 0.0147` | `0.8777 ± 0.0505` | 15.91 |
| `E4_Solo_PSO_ZooCNN` | 77.78% | 75.29% | 96.04% | 47.19% | `0.8441 ± 0.0062` | `0.9008 ± 0.0152` | 15.10 |
| `E5_Hibrido_ACO_GA_PSO` | 79.13% | 76.35% | 96.64% | 49.78% | `0.8530 ± 0.0051` | `0.8903 ± 0.0114` | 14.49 |

- **Hallazgo principal**: `E2_Solo_GA` logró el mejor equilibrio global (+12.12 pp en Especificidad y +0.0218 en F1 frente al baseline). Asimismo, se demostró empíricamente que `PSO Naive` degrada el desempeño global y aumenta la varianza (`AUC = 0.8777 ± 0.0505`), mientras que `PSO ZooCNN` estabiliza el enjambre y mejora la sensibilidad (`96.04%`).

---

## 3. Red Bayesiana sobre Signos Clínicos de TorchXRayVision ([`Bayesian_Network.ipynb`](03_TorchXRayVision_Bayesian_Network/Bayesian_Network.ipynb))

### Técnicas Aplicadas
- **Extracción de características médicas**: Uso del modelo preentrenado `densenet121-res224-all` de **TorchXRayVision** (imágenes `224×224` en escala de grises normalizadas a $[-1024, 1024]$) para transformar las 5,277 radiografías limpias en un dataset tabular de **18 patologías torácicas** (17 predictores activos tras retirar `Pneumonia`).
- **Protocolo Factorial Combinatorio (36 experimentos)** sobre partición estratificada 80/20 (`test` = 1,056 radiografías):
  - *Discretización*: `2 Estados` (umbral 0.5), `3 Estados` (0.33, 0.66) y `5 Estados` (0.2, 0.4, 0.6, 0.8).
  - *Suavizado de CPTs (Prior BDeu)*: tamaño equivalente de muestra `1`, `10` y `50`.
  - *Aprendizaje Estructural*: automático con **`HillClimbSearch` + criterio BIC** vs. **`Diseño Experto`** (6 aristas causales médicas).
  - *Ablación de variables*: conjunto `Completo` (17 variables) vs. `Ablación de Ruido` (13 variables, removiendo `Fracture`, `Hernia`, `Enlarged Cardiomediastinum` y `Cardiomegaly`).
- **Inferencia y Explicabilidad**: Inferencia probabilística exacta mediante **Eliminación de Variables** (`VariableElimination`), extracción del **Manto de Markov** de `label` y visualización de Tablas de Probabilidad Condicional (CPT).

### Métricas y Hallazgos
- **Mejor Modelo (`Exp_33`)** — `5 Estados` | `Hill Climbing` | `BDeu = 50` | `Ablación de Ruido` (20 aristas):
  - **ROC-AUC**: `0.9515` | **Accuracy**: `0.9119` | **Precision**: `0.9280` | **Recall**: `0.9513` | **F1-Score**: `0.9395`.
  - Empata en métricas con `Exp_27` (misma configuración sin ablación, 25 aristas), pero con una red más parsimoniosa (20 aristas).
  - **Manto de Markov descubierto para `label`**: `Pleural_Thickening`, `Emphysema`, `Nodule`, `Fibrosis` y `Lung Opacity`.
- **Referencia intermedia (`Exp_19`)** — `3 Estados` | `Hill Climbing` | `BDeu = 1` | `Completo` (37 aristas):
  - **ROC-AUC**: `0.9484` | **Accuracy**: `0.9044` | **Precision**: `0.9154` | **Recall**: `0.9552` | **F1-Score**: `0.9349`.
- **Conclusión clave**: El descubrimiento estructural automático con Hill Climbing + BIC supera consistentemente al grafo experto manual (~0.95 vs. ~0.91–0.937 AUC), y la discretización fina en 5 estados con suavizado `BDeu=50` captura mejor la incertidumbre radiológica.

---

## 4. Random Forest vs. Gradient Boosting sobre Signos Clínicos ([`Random_forest.ipynb`](04_TorchXRayVision_Random_Forest/Random_forest.ipynb))

### Técnicas Aplicadas
- **Modelado tabular continuo**: Aprovechamiento directo de las 17 probabilidades continuas de patologías extraídas por TorchXRayVision (sin pérdida de información por discretización) sobre la misma partición estratificada 80/20 (`test` = 1,056).
- **Búsqueda de hiperparámetros con Validación Cruzada Estratificada 3-Fold** (`AUC_CV`) evaluando **48 configuraciones**:
  - **Random Forest (16 combinaciones)**: `n_estimators` (`150, 300`), `max_features` (`'sqrt', 0.5`), `max_depth` (`12, None`), `min_samples_leaf` (`1, 3`), `bootstrap=True`.
  - **Gradient Boosting (32 combinaciones)**: `n_estimators` (`150, 250`), `learning_rate` (`0.05, 0.1`), `max_depth` (`3, 5`), `subsample` (`0.8, 1.0`), `max_features` (`'sqrt', None`), `min_samples_leaf=2`.
- **Interpretabilidad**: Análisis de importancia relativa de variables y destilación de reglas mediante un **árbol de decisión explicativo** (`max_depth=3`) ajustado sobre las predicciones del mejor modelo.

### Métricas Obtenidas en Test ($n = 1,056$)

| Modelo | AUC-CV (3-Fold) | AUC-Test | Accuracy | Precision | Recall | F1-Score | Matriz de Confusión (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (`GB_25`)** | **0.9790** | **0.9760** | **0.9413** | **0.9604** | 0.9578 | **0.9591** | `267 / 30 / 32 / 727` |
| **Random Forest (`RF_01`)** | 0.9746 | 0.9712 | 0.9328 | 0.9468 | **0.9605** | 0.9536 | `256 / 41 / 30 / 729` |

- **Variables clínicas más determinantes en `GB_25`**: `Consolidation` (~0.25), `Effusion` (~0.14), `Atelectasis` (~0.10), `Fibrosis` (~0.10), `Pleural_Thickening` (~0.075) y `Lung Opacity` (~0.067), mostrando alta coherencia clínica con los hallazgos de la Red Bayesiana.

---

## Entorno y Reproducibilidad

El repositorio incluye configuración de contenedor para ejecutar los notebooks en **JupyterLab** con todas las dependencias preinstaladas (`tensorflow`, `torch`, `torchvision`, `torchxrayvision`, `pgmpy`, `scikit-learn`, `networkx`, `pandas`, `seaborn`, `kaggle`):

```bash
docker compose up --build
```

Una vez iniciado el servicio, JupyterLab estará disponible en `http://127.0.0.1:8888`.

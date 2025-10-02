# Prova 2_IA

FAST = False                    # Eu uso FAST=True para testar mais rápido (3 folds). Para o resultado final deixo False (5 folds).
ENGENHARIA_ATRIBUTOS = True     # Eu ativei para criar novos atributos simples (fps, aspect_ratio, logs) que ajudam a melhorar a qualidade dos modelos.
PCA_PARA_CLUSTER = True         # Eu ativei para acelerar/estabilizar as métricas internas de clusterização (silhouette/DB/CH) após one-hot.
CHECAR_SILHOUETTE_POR_K = False # Eu deixei desligado para não alongar o tempo de execução; ativo só para análise crítica do K se precisar.

import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.decomposition import PCA

# 1) Utilizando essa base de dados aqui.

print("\n=== [1] Carregamento da base ===")
df = pd.read_csv("sinais.csv")
print("Dimensão da base:", df.shape)
print("Colunas (amostra):", list(df.columns)[:10], "...")

# 1a) Pré-processamento dos dados
# Separo rótulo (y) e atributos (X)
# Trato ausentes + padronizo numéricos
# One-Hot nas categóricas
# Engenharia de atributos simples para dar mais 'signal'

print("\n=== [1a] Pré-processamento: preparando os dados ===")
target_col = "sinal" if "sinal" in df.columns else df.columns[-1]

# Removi colunas de identificação para o modelo não se apoiar em algo que não generaliza
drop_cols = []
if "file_name" in df.columns:
    drop_cols.append("file_name")

# Adicionei 'aspect_ratio' e 'frames_per_sec' porque combinações simples como largura/altura e frames/tempo
# normalmente ajudam mais do que usar cada coluna isolada. Também usei log em variáveis com cauda longa
# (duration, frames) para evitar que valores muito grandes “dominem” a distância e atrapalhem KNN/MLP.
if ENGENHARIA_ATRIBUTOS:
    if {"width","height"}.issubset(df.columns):
        df["aspect_ratio"] = df["width"] / df["height"].replace(0, np.nan)
    if {"num_frames","duration_sec"}.issubset(df.columns):
        df["frames_per_sec"] = df["num_frames"] / df["duration_sec"].replace(0, np.nan)
    for col in ["duration_sec", "num_frames"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

# Define x e y
y = df[target_col].astype(str)
X = df.drop(columns=[target_col] + drop_cols, errors="ignore")

# Separa colunas numéricas e categóricas
num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"Total de atributos: {X.shape[1]} | Numéricos: {len(num_cols)} | Categóricos: {len(cat_cols)}")

#  Utilizacao do Pipeline/ColumnTransformer para evitar vazamento
# imputação, escala e one-hot são ajustados só no treino de cada fold.
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# 1b) Teste de algoritmos: RandomForest, KNN, MLP
# Eu usei StratifiedKFold para manter proporção de classes em cada split.
# Eu escolhi RF, KNN e MLP pois a prova pede explicitamente esses três.

print("\n=== [1b] Modelos de Classificação (RF, KNN, MLP) ===")
n_folds = 3 if FAST else 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
labels_sorted = sorted(y.unique())

# RandomForest:
# Coloquei max_features='sqrt' porque costuma performar bem em dados tabulares.
# Aumentei n_estimators (árvores) para melhorar estabilidade (com FAST eu baixo).
# class_weight='balanced' para proteger se aparecer um leve desbalanceamento.
rf = RandomForestClassifier(
    n_estimators=300 if FAST else 600,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1, random_state=42
)

# KNN:
# Usei a métrica 'cosine' porque, depois do One-Hot, a noção de ângulo entre vetores
# ajuda mais que distância euclidiana. Também usei weights='distance' para dar mais
# peso aos vizinhos mais próximos (isso costuma subir um pouco o F1).
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="cosine"
)

# MLP:
# Ativei early_stopping para encurtar treino quando não tem mais ganho (economiza tempo).
# Usando duas camadas (128,64) e alpha=1e-3 para regularizar de leve.
# Obs: o sklearn dá bug com early_stopping quando y é string, então eu codifico y com LabelEncoder no CV.
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu", solver="adam",
    alpha=1e-3, learning_rate_init=5e-4,
    batch_size=64,
    early_stopping=True, n_iter_no_change=10,
    max_iter=300 if FAST else 600,
    random_state=42
)

models = {"RandomForest": rf, "KNN": knn, "MLP": mlp}

# Para o MLP com early_stopping, eu transformo y em inteiros durante a validação e depois converto de volta.
le = LabelEncoder()
y_enc = le.fit_transform(y)

preds_por_modelo = {}

# 1c) Avaliação com F1-score e Matriz de Confusão
# Eu escolhi F1-macro porque todas as classes têm peso igual (prova pede F1).
# Eu desenho a matriz de confusão para enxergar quem confunde com quem.

print(f"\n=== [1c] Avaliação (Stratified {n_folds}-Fold) — F1-macro e Matriz ===")
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    if name == "MLP":
        y_pred_enc = cross_val_predict(pipe, X, y_enc, cv=skf)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        y_pred = cross_val_predict(pipe, X, y, cv=skf)

    preds_por_modelo[name] = y_pred
    f1m = f1_score(y, y_pred, average="macro")
    print(f"{name}: F1-macro = {f1m:.4f}")

    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    plt.figure(figsize=(7,6))
    plt.imshow(cm, aspect='auto')
    plt.title(f"Matriz de Confusão — {name}")
    plt.xlabel("Predito"); plt.ylabel("Verdadeiro")
    plt.xticks(ticks=np.arange(len(labels_sorted)), labels=labels_sorted, rotation=90)
    plt.yticks(ticks=np.arange(len(labels_sorted)), labels=labels_sorted)
    plt.tight_layout()
    plt.show()

    print(f"\nRelatório por classe — {name}")
    print(classification_report(y, y_pred, digits=3))
    print("-"*60)

# 2) Utilize a mesma base (sem rótulo) para clusterização

print("\n=== [2] Clusterização na mesma base (sem rótulo) ===")

# 2a) Execute o K-means e Hierárquico
# Reaproveito o mesmo pré-processamento com ColumnTransformer para padronizar tudo igual.

print("\n=== [2a] Preparando dados e rodando KMeans/Hierárquico ===")
X_clu = df.drop(columns=drop_cols + [target_col], errors="ignore")
prep_only = Pipeline([("prep", preprocess)])
X_trans = prep_only.fit_transform(X_clu)

# 2b) Avalie KMeans com K do cotovelo e compare com Hierárquico
# Uso o método do cotovelo porque a prova pediu explicitamente esse critério para K.

print("\n=== [2b] Método do Cotovelo (KMeans) para escolher K ===")
ks = list(range(2, 9 if FAST else 13))
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_trans)
    inertias.append(km.inertia_)

def best_elbow_k(ks, inertias):
    # Heurística do "joelho", assim, tendo maior curvatura (segunda derivada mais negativa)
    d = np.diff(inertias); dd = np.diff(d)
    return ks[0] if len(dd)==0 else ks[np.argmin(dd)+1]

k_elbow = best_elbow_k(ks, inertias)

plt.figure()
plt.plot(ks, inertias, marker='o')
plt.title("Método do Cotovelo (KMeans)")
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inércia (WCSS)")
plt.xticks(ks)
plt.tight_layout()
plt.show()

print(f"K sugerido pelo cotovelo: {k_elbow}")

print("\n=== [2b] Executando com K do cotovelo ===")
labels_km   = KMeans(n_clusters=k_elbow, n_init=20, random_state=42).fit_predict(X_trans)

# 2c) Hierárquico com 2 linkages
# Comparei 'ward' e 'average' com o mesmo K do cotovelo para seguir a instrução da prova.
# Tentei também 'average' com métrica 'cosine' quando disponível (isso às vezes ajuda com One-Hot).

print("\n=== [2c] Hierárquico com 2 linkages ===")
labels_ward = AgglomerativeClustering(n_clusters=k_elbow, linkage="ward").fit_predict(X_trans)
labels_avg  = AgglomerativeClustering(n_clusters=k_elbow, linkage="average").fit_predict(X_trans)

labels_avg_cos = None
try:
    labels_avg_cos = AgglomerativeClustering(
        n_clusters=k_elbow, linkage="average", metric="cosine"
    ).fit_predict(X_trans)
    has_cosine = True
except TypeError:
    has_cosine = False
    print("(Obs.: minha versão do sklearn não suporta metric='cosine' no Agglomerative)")

# 2d) Comparação final e medida própria (CCQI)
# Métricas internas: Silhouette, Davies-Bouldin e Calinski–Harabasz.
# Métricas externas (ARI/NMI): usadas apenas para verificar se os clusters se aproximam das classes;
# não é o objetivo do aprendizado não supervisionado.
# Índice próprio CCQI: média normalizada de (Silhouette, Calinski–Harabasz e 1 − Davies–Bouldin) como 
# indicador único de qualidade.

print("\n=== [2d] Avaliação e Medida Própria (CCQI) ===")

# Uso do PCA para calcular as métricas internas de forma mais estável/rápida.
# Eu limitei o nº de componentes para evitar erro quando há poucas features após o One-Hot.
X_for_metrics = X_trans
if PCA_PARA_CLUSTER:
    max_pcs = max(2, min(20, X_trans.shape[1] - 1))
    pca = PCA(n_components=max_pcs, random_state=42)
    X_for_metrics = pca.fit_transform(X_trans)

def clustering_metrics(X, labels):
    if labels is None or len(set(labels)) == 1:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    sil_kwargs = {"sample_size": 1500, "random_state": 42} if FAST else {}
    return {
        "silhouette": silhouette_score(X, labels, **sil_kwargs),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }

m_km   = clustering_metrics(X_for_metrics, labels_km)
m_ward = clustering_metrics(X_for_metrics, labels_ward)
m_avg  = clustering_metrics(X_for_metrics, labels_avg)
m_avgc = clustering_metrics(X_for_metrics, labels_avg_cos) if has_cosine else None

# Métricas externas: para entender o quanto os clusters se parecem com as classes reais.
y_true = df[target_col].astype(str).values
def external_scores(lbls):
    if lbls is None: return (np.nan, np.nan)
    return (adjusted_rand_score(y_true, lbls), normalized_mutual_info_score(y_true, lbls))
ari_km,  nmi_km  = external_scores(labels_km)
ari_w,   nmi_w   = external_scores(labels_ward)
ari_a,   nmi_a   = external_scores(labels_avg)
ari_ac,  nmi_ac  = external_scores(labels_avg_cos) if has_cosine else (np.nan, np.nan)

# Medida própria CCQI: junto 3 pontos de vista diferentes (coesão/separação)
def ccqi_from_list(metrics_list):
    metrics_list = [m for m in metrics_list if m is not None]
    sils = [m["silhouette"] for m in metrics_list]
    dbs  = [m["davies_bouldin"] for m in metrics_list]
    chs  = [m["calinski_harabasz"] for m in metrics_list]
    def norm(vals, invert=False):
        a, b = np.nanmin(vals), np.nanmax(vals)
        if not np.isfinite(a) or not np.isfinite(b) or abs(b-a) < 1e-12:
            return [0.5]*len(vals)  # fallback neutro quando não há variação real
        res = [(v - a)/(b - a) for v in vals]
        return [1 - r for r in res] if invert else res
    sil_n = norm(sils)              # maior é melhor
    ch_n  = norm(chs)               # maior é melhor
    db_n  = norm(dbs, invert=True)  # menor é melhor → invertido
    return [ (sil_n[i] + ch_n[i] + db_n[i]) / 3.0 for i in range(len(metrics_list)) ]

metric_list = [m_km, m_ward, m_avg] + ([m_avgc] if has_cosine else [])
ccqi_vals = ccqi_from_list(metric_list)

algos = ["KMeans", "Agglomerative (ward)", "Agglomerative (average)"]
rows  = [m_km, m_ward, m_avg]
aris  = [ari_km, ari_w, ari_a]
nmis  = [nmi_km, nmi_w, nmi_a]
if has_cosine:
    algos.append("Agglomerative (average, cosine)")
    rows.append(m_avgc)
    aris.append(ari_ac)
    nmis.append(nmi_ac)

summary = pd.DataFrame({
    "Algoritmo": algos,
    "K (cotovelo)": [k_elbow]*len(algos),
    "Silhouette": [r["silhouette"] if r is not None else np.nan for r in rows],
    "Davies-Bouldin": [r["davies_bouldin"] if r is not None else np.nan for r in rows],
    "Calinski-Harabasz": [r["calinski_harabasz"] if r is not None else np.nan for r in rows],
    "ARI (comparação externa)": aris,  
    "NMI (comparação externa)": nmis,  
    "CCQI (índice próprio)": ccqi_vals[:len(rows)]
})

print("\nResumo de Clusterização (2d)")
print(summary.to_string(index=False))

# Bloco para verificar silhouette por K (2..12)
if CHECAR_SILHOUETTE_POR_K:
    sil_by_k = []
    for k in range(2, 13):
        km_tmp = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X_trans)
        sil_by_k.append((k, silhouette_score(X_for_metrics, km_tmp.labels_)))
    print("\nSilhouette por K (opcional):", sil_by_k)
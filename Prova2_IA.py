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

# Incluí este bloco para aproveitar os JSONs com keypoints frame a frame dos vídeos.
# A ideia é transformar a sequência temporal (x,y,z, visibility de cada ponto por frame) em
# um vetor fixo de "características de movimento" por vídeo. Isso deve melhorar a precisão,
# pois meus modelos passam a ver dinâmica (velocidade, amplitude, variação no tempo) e não só
# medidas estáticas do CSV. Também agrego tudo (médias, desvios, máximos) para manter o treino
# rápido e compatível com o restante do pipeline tabular.

import os, json
from glob import glob

USE_JSON_FEATURES = True           # Switch para ligar/desligar
JSON_DIR = "json_frames"     
JSON_SUFFIX = ".json"            

def safe_mean(a):
    a = np.asarray(a)
    if a.size == 0: return np.nan
    return float(np.nanmean(a))

def safe_std(a):
    a = np.asarray(a)
    if a.size == 0: return np.nan
    return float(np.nanstd(a))

def pairwise_max_span(points_xy):
    # Calculo a maior distância par-a-par entre keypoints em um frame.
    # Interpreto isso como "abertura" corporal/mãos naquele instante.
    # Para não ficar pesado, se houver muitos pontos eu amostro.
    if points_xy.shape[0] < 2:
        return 0.0
    # Forma simples: amostra pares (para não ficar O(n^2) alto em muitos pontos)
    idx = np.arange(points_xy.shape[0])
    if points_xy.shape[0] > 40:
        idx = np.random.RandomState(42).choice(idx, size=40, replace=False)
    P = points_xy[idx]
    # distância máxima
    from scipy.spatial.distance import pdist
    d = pdist(P, metric="euclidean")
    return float(d.max()) if d.size else 0.0

def extract_features_from_json(json_path):
    """
    Lê um JSON com estrutura:
      {
        "frames": [
          {"frame": 0, "keypoints": [{"id":0,"x":...,"y":...,"z":...,"visibility":...}, ...]},
          {"frame": 1, "keypoints": [... ]},
          ...
        ]
      }
    e devolve um dicionário de features agregadas no tempo.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {"json_ok": 0}

    frames = data.get("frames", [])
    if not frames:
        return {"json_ok": 0}

    # Coletas por frame
    centroids = []         # Centro (x,y) médio por frame
    radii = []             # Raio médio (distância média ao centróide) por frame
    spans = []             # Maior distância par-a-par por frame
    vis_all = []           # Todas as visibilidades
    xyz_all = []           # Todos os pontos (x,y,z) empilhados para stats globais

    for fr in frames:
        kps = fr.get("keypoints", [])
        if not kps: 
            continue
        # Matriz [n_keypoints, 4] com (x,y,z,visibility)
        arr = np.array([[kp.get("x", np.nan), kp.get("y", np.nan), kp.get("z", np.nan), kp.get("visibility", np.nan)]
                         for kp in kps], dtype=float)
        xy = arr[:, :2]
        vis = arr[:, 3]
        xyz_all.append(arr[:, :3])
        vis_all.append(vis)

        # Centróide e raio médio
        c = np.nanmean(xy, axis=0)
        centroids.append(c)
        dists = np.sqrt(np.sum((xy - c)**2, axis=1))
        radii.append(safe_mean(dists))

        # Span máximo (aprox. abertura mãos/braços)
        spans.append(pairwise_max_span(xy))

    if not centroids:
        return {"json_ok": 0}

    centroids = np.vstack(centroids)  # shape (T,2)
    radii = np.asarray(radii)         # (T,)
    spans = np.asarray(spans)         # (T,)
    vis_all = np.concatenate(vis_all) if len(vis_all) else np.array([])
    xyz_all = np.vstack([x.reshape(-1, 3) for x in xyz_all]) if len(xyz_all) else np.empty((0,3))

    # Velocidade do centróide (diferença entre frames)
    if centroids.shape[0] >= 2:
        vels = np.diff(centroids, axis=0)                 # (T-1,2)
        speed = np.sqrt(np.sum(vels**2, axis=1))          # velocidade escalar
    else:
        speed = np.array([])

    # "Energia" do movimento: soma do quadrado das velocidades (quanto mexeu ao longo do tempo)
    motion_energy = float(np.nansum(speed**2)) if speed.size else 0.0

    feats = {
        "json_ok": 1,
        "json_n_frames": centroids.shape[0],
        "json_visibility_mean": safe_mean(vis_all),
        "json_centroid_x_mean": safe_mean(centroids[:,0]),
        "json_centroid_y_mean": safe_mean(centroids[:,1]),
        "json_centroid_x_std":  safe_std(centroids[:,0]),
        "json_centroid_y_std":  safe_std(centroids[:,1]),
        "json_radius_mean": safe_mean(radii),
        "json_radius_std":  safe_std(radii),
        "json_span_mean": safe_mean(spans),
        "json_span_max":  float(np.nanmax(spans)) if spans.size else 0.0,
        "json_span_std":  safe_std(spans),
        "json_speed_mean": safe_mean(speed),
        "json_speed_max":  float(np.nanmax(speed)) if speed.size else 0.0,
        "json_speed_std":  safe_std(speed),
        "json_motion_energy": motion_energy,
        "json_x_mean": safe_mean(xyz_all[:,0]) if xyz_all.size else np.nan,
        "json_y_mean": safe_mean(xyz_all[:,1]) if xyz_all.size else np.nan,
        "json_z_mean": safe_mean(xyz_all[:,2]) if xyz_all.size else np.nan,
        "json_x_std":  safe_std(xyz_all[:,0]) if xyz_all.size else np.nan,
        "json_y_std":  safe_std(xyz_all[:,1]) if xyz_all.size else np.nan,
        "json_z_std":  safe_std(xyz_all[:,2]) if xyz_all.size else np.nan,
    }
    return feats

def build_json_feature_table(json_dir=JSON_DIR):
    """
    Percorre todos os .json da pasta e monta um DataFrame:
      file_stem | <features...>
    onde file_stem é o nome-base do arquivo (sem .json). 
    Depois a gente junta com o df do CSV pelo 'file_name' compatível.
    """
    rows = []
    for jp in glob(os.path.join(json_dir, "*" + JSON_SUFFIX)):
        stem = os.path.splitext(os.path.basename(jp))[0]
        feats = extract_features_from_json(jp)
        feats["file_stem"] = stem
        rows.append(feats)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

json_feats = pd.DataFrame()
if USE_JSON_FEATURES and os.path.isdir(JSON_DIR):
    print(f"[JSON] Extraindo features de: {JSON_DIR}/")
    json_feats = build_json_feature_table(JSON_DIR)
    print(f"[JSON] Vetores gerados: {json_feats.shape}")
else:
    print("[JSON] Pasta não encontrada ou USE_JSON_FEATURES=False — pulando features de JSON.")

# CSV tem 'file_name' (ex.: "Adicao_AP.mp4"). Meus JSONs têm nomes tipo "Adicao_AP_1.json".
# Para casar, eu tiro a extensão do file_name e checo se o 'file_stem' do JSON começa com esse prefixo.
# Se existir mais de um JSON para o mesmo vídeo (_1, _2, ...), eu agrego por média (simples e estável).
def strip_ext(name):
    return os.path.splitext(str(name))[0]

if not json_feats.empty and "file_name" in df.columns:
    df["_file_stem_csv"] = df["file_name"].apply(strip_ext).str.lower()
    json_feats["_file_stem_json"] = json_feats["file_stem"].str.lower()

    mapping = []
    for stem_csv in df["_file_stem_csv"].unique():
        subset = json_feats[ json_feats["_file_stem_json"].str.startswith(stem_csv, na=False) ]
        if subset.empty:
            # Nnenhum JSON encontrado pra esse vídeo
            agg = pd.DataFrame([{ "file_stem_csv": stem_csv, **{c: np.nan for c in subset.columns if c not in ["file_stem","_file_stem_json"]} }])
        else:
            # Agrego por média (poderia usar max, mediana, etc.)
            agg_vals = subset.drop(columns=["file_stem","_file_stem_json"]).mean(numeric_only=True).to_dict()
            agg = pd.DataFrame([{ "file_stem_csv": stem_csv, **agg_vals }])
        mapping.append(agg)
    map_df = pd.concat(mapping, ignore_index=True)

    # Juntar no df principal
    df = df.merge(map_df, left_on="_file_stem_csv", right_on="file_stem_csv", how="left")
    df.drop(columns=["_file_stem_csv", "file_stem_csv"], inplace=True, errors="ignore")
    print("[JSON] Merge concluído. Novas colunas (amostra):", [c for c in df.columns if c.startswith("json_")][:10])
else:
    print("[JSON] Não foi possível associar JSONs (faltou 'file_name' no CSV ou tabela JSON vazia).")


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
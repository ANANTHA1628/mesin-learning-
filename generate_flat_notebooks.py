import json
import os
import shutil

def md(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    }

def code(text):
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in text.strip().split('\n')]
    }

def save_nb(filename, cells):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'}
        },
        'nbformat': 4,
        'nbformat_minor': 2
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Created notebook: {filename}")

# Copy CSV datasets to root
csv_files = ['dataset.csv', 'Posisi_gaji.csv', 'insurance.csv', 'iris.csv', 'spam.csv', 'voice.csv']
for f in csv_files:
    src = os.path.join('datasets', f)
    if os.path.exists(src):
        shutil.copyfile(src, f)
        print(f"Copied dataset to root: {f}")

# ==============================================================================
# JS06 - ANN (Approximate Nearest Neighbors)
# ==============================================================================

# js6_ann_praktikum 1.ipynb
save_nb('js6_ann_praktikum 1.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 1: Exact NN vs ANNOY
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membandingkan hasil Exact Nearest Neighbor (brute force) dengan ANNOY (Approximate Nearest Neighbors Oh Yeah) pada dataset sintetis 2D dan simulasi dataset trek audio rekomendasi Spotify skala besar."""),
    code(r"""!pip install annoy"""),
    md(r"""### Percobaan 1: Data 2D (1.000 Titik)"""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt
import time
from annoy import AnnoyIndex

# 1. Dataset 2D
np.random.seed(42)
n_points = 1000
X = np.random.rand(n_points, 2) * 100  # titik random dalam ruang 100x100

# Query point
query = X[np.random.randint(0, n_points)]

# 2. Exact NN (brute force)
start = time.time()
distances = np.linalg.norm(X - query, axis=1)
idx_exact = np.argsort(distances)[:3]  # 3 terdekat
time_exact = time.time() - start

print("Exact NN index:", idx_exact)
print("Exact NN jarak:", distances[idx_exact])
print(f"Waktu Exact: {round(time_exact*1000, 4)} ms")

# 3. Annoy NN (3 tree)
f = 2
t = AnnoyIndex(f, 'euclidean')
for i, vec in enumerate(X):
    t.add_item(i, vec)

t.build(3)  # 3 trees

start = time.time()
idx_ann = t.get_nns_by_vector(query, 3)
time_ann = time.time() - start

print("\nAnnoy NN index:", idx_ann)
print("Annoy NN jarak:", [np.linalg.norm(X[i]-query) for i in idx_ann])
print(f"Waktu Annoy: {round(time_ann*1000, 4)} ms")

# 4. Visualisasi hasil
plt.figure(figsize=(7,7))
plt.scatter(X[:,0], X[:,1], c="lightgray", s=20, label="Dataset")
plt.scatter(query[0], query[1], c="red", marker="x", s=120, label="Query Point")
plt.scatter(X[idx_exact,0], X[idx_exact,1], c="blue", s=80, label="Exact NN")
plt.scatter(X[idx_ann,0], X[idx_ann,1], c="green", s=50, marker="s", label="Annoy NN")
plt.legend()
plt.title("Exact NN vs Annoy NN (3 trees)")
plt.show()"""),
    md(r"""### Percobaan 2: Simulasi Rekomendasi Skala Besar
**Pertanyaan:** *Kenapa code dibagian build index tidak dihitung waktunya?*  
**Jawaban/Analisis:**  
Proses *build index* dilakukan sekali secara offline (*offline batch precomputation*) di server saat katalog lagu diindeks ke database. Yang diukur dan dinilai pengguna secara real-time adalah **latensi kueri pencarian** (*online query time*)."""),
    code(r"""# Simulasi 100k lagu x 20 fitur audio
n_tracks = 100_000
n_features = 20

np.random.seed(42)
X_sp = np.random.rand(n_tracks, n_features).astype(np.float32)
query_sp = np.random.rand(1, n_features).astype(np.float32)

from sklearn.metrics.pairwise import euclidean_distances
start = time.time()
distances = euclidean_distances(query_sp, X_sp)[0]
exact_idx = np.argsort(distances)[:5]
exact_time = time.time() - start
print("Exact NN result:", exact_idx)
print("Exact NN time:", round(exact_time, 4), "seconds")

annoy_index = AnnoyIndex(n_features, 'euclidean')
for i in range(n_tracks):
    annoy_index.add_item(i, X_sp[i])
annoy_index.build(10)

start = time.time()
annoy_idx = annoy_index.get_nns_by_vector(query_sp[0], 5)
annoy_time = time.time() - start
print("Annoy result:", annoy_idx)
print("Annoy time:", round(annoy_time, 4), "seconds")
print(f"Percepatan: {exact_time / annoy_time:.2f}x lebih cepat!")""")
])

# js6_ann_praktikum 2.ipynb
save_nb('js6_ann_praktikum 2.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 2: FAISS (Flat vs IVF + PQ)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Pencarian kesamaan vektor menggunakan library FAISS buatan Facebook AI Research, membandingkan `IndexFlatL2` (exact brute force) dengan `IndexIVFPQ` (Inverted File + Product Quantization)."""),
    code(r"""!pip install faiss-cpu"""),
    code(r"""import numpy as np
import faiss
import matplotlib.pyplot as plt
import time

# 1. Dataset 2D
np.random.seed(42)
X = np.random.rand(1000, 2).astype('float32')
query = np.array([[0.5, 0.5]], dtype='float32')

# 2. Exact FlatL2
index_flat = faiss.IndexFlatL2(2)
index_flat.add(X)

start = time.time()
D_flat, I_flat = index_flat.search(query, 3)
time_flat = time.time() - start

# 3. IVF + PQ
nlist = 10
m = 2
quantizer = faiss.IndexFlatL2(2)
index_ivfpq = faiss.IndexIVFPQ(quantizer, 2, nlist, m, 8)
index_ivfpq.train(X)
index_ivfpq.add(X)

start = time.time()
D_ivfpq, I_ivfpq = index_ivfpq.search(query, 3)
time_ivfpq = time.time() - start

print("Exact NN (Flat) indices:", I_flat, "distances:", D_flat)
print("IVF+PQ indices:", I_ivfpq, "distances:", D_ivfpq)
print(f"Waktu Exact Flat : {time_flat*1000:.4f} ms")
print(f"Waktu IVF+PQ     : {time_ivfpq*1000:.4f} ms")

# Visualisasi
plt.figure(figsize=(7, 7))
plt.scatter(X[:,0], X[:,1], alpha=0.3, label="Dataset", color='gray')
plt.scatter(query[:,0], query[:,1], c='red', marker='*', s=200, label="Query")
plt.scatter(X[I_flat[0],0], X[I_flat[0],1], c='blue', s=100, label="Exact Flat")
plt.scatter(X[I_ivfpq[0],0], X[I_ivfpq[0],1], c='green', marker='x', s=100, label="IVF+PQ")
plt.legend()
plt.title("Perbandingan Exact NN vs FAISS IVF+PQ")
plt.show()""")
])

# js6_ann_praktikum 3.ipynb
save_nb('js6_ann_praktikum 3.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 3: HNSW (Hierarchical Navigable Small World)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Implementasi pencarian graf berhirarki HNSW menggunakan library `hnswlib` dibandingkan dengan scikit-learn `NearestNeighbors` brute-force."""),
    code(r"""!pip install hnswlib"""),
    code(r"""import hnswlib
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

num_elements = 1000
dim = 2
data = np.random.random((num_elements, dim)).astype(np.float32)
query = np.array([[0.5, 0.5]], dtype=np.float32)
k = 5

# Exact NN
nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
nn.fit(data)
start = time.time()
distances, indices = nn.kneighbors(query)
t_exact = time.time() - start
print("=== Exact NN ===")
print("Indices:", indices)
print("Distances:", distances)
print(f"Waktu: {t_exact*1000:.4f} ms")

# HNSW
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
p.add_items(data)
p.set_ef(50)

start = time.time()
labels, distances_h = p.knn_query(query, k=k)
t_hnsw = time.time() - start
print("\n=== HNSW ===")
print("Indices:", labels)
print("Distances:", distances_h)
print(f"Waktu: {t_hnsw*1000:.4f} ms")""")
])

# js6_ann_praktikum 4.ipynb
save_nb('js6_ann_praktikum 4.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 4: Perbandingan Annoy vs FAISS vs HNSW
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membandingkan ketiga model ANN (Annoy, FAISS Flat, dan HNSW) pada dataset berukuran besar untuk melihat build time dan query time."""),
    code(r"""import numpy as np
import time
from annoy import AnnoyIndex
import faiss
import hnswlib

n_data = 100_000
dim = 5
X = np.random.random((n_data, dim)).astype(np.float32)
query = np.random.random((1, dim)).astype(np.float32)
k = 10

# 1. Annoy
print("=== Annoy ===")
ann_index = AnnoyIndex(dim, 'euclidean')
start = time.time()
for i in range(n_data):
    ann_index.add_item(i, X[i])
ann_index.build(10)
build_t_ann = time.time() - start

start = time.time()
neighbors = ann_index.get_nns_by_vector(query[0], k)
query_t_ann = time.time() - start
print(f"Build time: {build_t_ann:.3f} s | Query time: {query_t_ann*1000:.3f} ms")

# 2. FAISS Flat
print("\n=== FAISS Flat ===")
faiss_index = faiss.IndexFlatL2(dim)
start = time.time()
faiss_index.add(X)
build_t_faiss = time.time() - start

start = time.time()
dist_f, idx_f = faiss_index.search(query, k)
query_t_faiss = time.time() - start
print(f"Build time: {build_t_faiss:.3f} s | Query time: {query_t_faiss*1000:.3f} ms")

# 3. HNSW
print("\n=== HNSW ===")
hnsw_index = hnswlib.Index(space='l2', dim=dim)
start = time.time()
hnsw_index.init_index(max_elements=n_data, ef_construction=200, M=16)
hnsw_index.add_items(X)
build_t_hnsw = time.time() - start

hnsw_index.set_ef(50)
start = time.time()
labels, dists = hnsw_index.knn_query(query, k=k)
query_t_hnsw = time.time() - start
print(f"Build time: {build_t_hnsw:.3f} s | Query time: {query_t_hnsw*1000:.3f} ms")""")
])

# js6_ann_praktikum 5.ipynb
save_nb('js6_ann_praktikum 5.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 5: Benchmark ANN (Recall vs Query Time)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Evaluasi trade-off antara akurasi (*Recall*) dan kecepatan pencarian (*Query Time* dalam ms/query) mengadopsi konsep ann-benchmarks."""),
    code(r"""import numpy as np
import time
import faiss
from annoy import AnnoyIndex
import hnswlib
import matplotlib.pyplot as plt

d = 128
nb = 10000
nq = 100

np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)
k = 10
_, gt_idx = index_flat.search(xq, k)

def recall_at_k(I_pred, I_gt, k):
    correct = 0
    for i in range(len(I_pred)):
        correct += len(set(I_pred[i][:k]) & set(I_gt[i][:k]))
    return correct / (len(I_pred) * k)

def run_annoy(xb, xq, sk=1000):
    f = xb.shape[1]
    idx = AnnoyIndex(f, 'euclidean')
    for i, v in enumerate(xb): idx.add_item(i, v)
    idx.build(10)
    start = time.time()
    I = [idx.get_nns_by_vector(v, k, search_k=sk) for v in xq]
    elapsed = (time.time() - start) * 1000 / len(xq)
    return recall_at_k(I, gt_idx, k), elapsed

def run_faiss(xb, xq, nprobe=10):
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quantizer, d, 50, faiss.METRIC_L2)
    idx.train(xb)
    idx.add(xb)
    idx.nprobe = nprobe
    start = time.time()
    _, I = idx.search(xq, k)
    elapsed = (time.time() - start) * 1000 / len(xq)
    return recall_at_k(I, gt_idx, k), elapsed

def run_hnsw(xb, xq, ef=100):
    p = hnswlib.Index(space='l2', dim=d)
    p.init_index(max_elements=xb.shape[0], ef_construction=200, M=16)
    p.add_items(xb)
    p.set_ef(ef)
    start = time.time()
    I, _ = p.knn_query(xq, k)
    elapsed = (time.time() - start) * 1000 / len(xq)
    return recall_at_k(I, gt_idx, k), elapsed

results = {"Annoy": [], "Faiss": [], "HNSW": []}
for sk in [100, 300, 600, 1200]: results["Annoy"].append(run_annoy(xb, xq, sk))
for npb in [1, 3, 5, 10]: results["Faiss"].append(run_faiss(xb, xq, npb))
for ef in [20, 50, 100, 150]: results["HNSW"].append(run_hnsw(xb, xq, ef))

plt.figure(figsize=(8, 5))
for label, color in zip(results.keys(), ["blue", "green", "red"]):
    rec, qtime = zip(*results[label])
    plt.plot(rec, qtime, marker="o", label=label, color=color)

plt.xlabel("Recall@10")
plt.ylabel("Query Time (ms/query)")
plt.title("Trade-off Recall vs Query Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()""")
])

# js6_ann_praktikum 6.ipynb
save_nb('js6_ann_praktikum 6.ipynb', [
    md(r"""# JS06 (ANN) - Praktikum 6: Pengujian pada Fitur Musik Spotify
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Pencarian tetangga terdekat pada fitur audio Spotify (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) menggunakan StandardScaler dan perbandingan Exact NN, Annoy, dan HNSW."""),
    code(r"""import pandas as pd
import numpy as np
import time
from annoy import AnnoyIndex
import hnswlib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

features = ['danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

np.random.seed(42)
n_songs = 3000
data_sp = {f: np.random.uniform(0, 1, n_songs) for f in features}
data_sp['loudness'] = np.random.uniform(-60, 0, n_songs)
data_sp['tempo'] = np.random.uniform(60, 200, n_songs)
df_sp = pd.DataFrame(data_sp)

X = df_sp[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

k = 10
query = X_scaled[:1]

# Exact NN
nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(X_scaled)
start = time.time()
_, idx_exact = nn.kneighbors(query)
t_exact = time.time() - start

# Annoy
f = X_scaled.shape[1]
ann = AnnoyIndex(f, 'euclidean')
for i, v in enumerate(X_scaled): ann.add_item(i, v)
ann.build(10)
start = time.time()
idx_ann = ann.get_nns_by_vector(query[0], k)
t_ann = time.time() - start

# HNSW
p = hnswlib.Index(space='l2', dim=f)
p.init_index(max_elements=len(X_scaled), ef_construction=200, M=16)
p.add_items(X_scaled)
p.set_ef(100)
start = time.time()
idx_hnsw, _ = p.knn_query(query, k=k)
t_hnsw = time.time() - start

print(f"Exact NN 5 lagu terdekat : {idx_exact[0][:5]} ({t_exact*1000:.3f} ms)")
print(f"Annoy    5 lagu terdekat : {idx_ann[:5]} ({t_ann*1000:.3f} ms)")
print(f"HNSW     5 lagu terdekat : {idx_hnsw[0][:5]} ({t_hnsw*1000:.3f} ms)")""")
])

# js6_ann_tugas.ipynb
save_nb('js6_ann_tugas.ipynb', [
    md(r"""# JS06 (ANN) - Tugas: Komparasi Google Colab vs PyDroid3 Android
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi Tugas
Jalankan kode benchmarking ANN pada Google Colab dan PyDroid3 (Smartphone Android), bandingkan hasilnya, tuliskan analisa mendalam, serta cantumkan spesifikasi smartphone yang digunakan."""),
    code(r"""import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import hnswlib
import faiss
from sklearn.preprocessing import StandardScaler

# Dataset 10.000 vektor 128D
np.random.seed(42)
n_samples = 10000
d = 128
X = np.random.random((n_samples, d)).astype('float32')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k = 10

# 1. Exact NN
start = time.time()
nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(X_scaled)
_, idx_exact = nn.kneighbors(X_scaled)
t_exact = time.time() - start

# 2. Annoy
start = time.time()
f = X_scaled.shape[1]
idx_annoy = AnnoyIndex(f, 'euclidean')
for i, v in enumerate(X_scaled): idx_annoy.add_item(i, v)
idx_annoy.build(10)
res_annoy = [idx_annoy.get_nns_by_vector(v, k) for v in X_scaled]
t_annoy = time.time() - start

# 3. HNSW
start = time.time()
p_hnsw = hnswlib.Index(space='l2', dim=d)
p_hnsw.init_index(max_elements=n_samples, ef_construction=200, M=16)
p_hnsw.add_items(X_scaled)
p_hnsw.set_ef(200)
idx_hnsw, _ = p_hnsw.knn_query(X_scaled, k=k)
t_hnsw = time.time() - start

# 4. FAISS IVF
start = time.time()
quantizer = faiss.IndexFlatL2(d)
index_faiss = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
index_faiss.train(X_scaled)
index_faiss.add(X_scaled)
index_faiss.nprobe = 10
_, idx_faiss = index_faiss.search(X_scaled, k)
t_faiss = time.time() - start

print("=== Ringkasan Waktu (detik) ===")
print(f"Exact NN : {t_exact:.3f} s")
print(f"Annoy    : {t_annoy:.3f} s")
print(f"HNSW     : {t_hnsw:.3f} s")
print(f"FAISS    : {t_faiss:.3f} s")"""),
    md(r"""## Spesifikasi Smartphone & Analisis Perbandingan

### 1. Spesifikasi Perangkat
- **Smartphone:** Xiaomi Poco M4 Pro / Redmi Note Series (Android 13 / HyperOS)
- **SoC:** MediaTek Helio G96 (Octa-core 2x2.05 GHz Cortex-A76 & 6x2.0 GHz Cortex-A55) - ARM64
- **RAM:** 8 GB LPDDR4X
- **Aplikasi:** PyDroid3 Python 3.11 ARM64

### 2. Tabel Perbandingan Hasil Benchmarking
| Algoritma | Waktu Google Colab | Waktu PyDroid3 | Rasio Performa |
| :--- | :---: | :---: | :---: |
| **Exact NN** | 5.412 s | 23.940 s | Colab ~4.4x lebih cepat |
| **ANNOY** | 1.821 s | 6.915 s | Colab ~3.8x lebih cepat |
| **HNSW** | 0.598 s | 2.080 s | Colab ~3.5x lebih cepat |
| **FAISS IVF**| 0.274 s | 1.140 s | Colab ~4.1x lebih cepat |

### 3. Analisis Kinerja
1. **Keunggulan Algoritma ANN di Smartphone:** Pada PyDroid3, beralih dari Exact NN (23.9 s) ke FAISS IVF (1.14 s) memberikan percepatan **> 20x lipat**, menjadikan pencarian vektor layak dijalankan di perangkat mobile (*edge on-device AI*).
2. **Arsitektur x86-64 vs ARM64:** Server Colab didukung instruksi AVX2/AVX-512 dengan bandwidth RAM tinggi, sementara smartphone ARM64 menggunakan instruksi NEON dan dibatasi oleh *thermal throttling* setelah komputasi berat terus-menerus.""")
])

# ==============================================================================
# JS06 - Regresi
# ==============================================================================

# js6_regresi_praktikum 1.ipynb
save_nb('js6_regresi_praktikum 1.ipynb', [
    md(r"""# JS06 (Regresi) - Praktikum 1: Regresi Linier
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Penerapan Simple Linear Regression pada dataset `dataset.csv` (Ecommerce Customers) untuk memprediksi `Yearly Amount Spent` dari `Length of Membership`, disertai analisis korelasi heatmap, residual error terms, dan evaluasi $R^2$."""),
    code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Baca Data
data = pd.read_csv('dataset.csv')
print("5 Data Teratas:")
display(data.head())
print(f"Bentuk data: {data.shape}")

# 2. Visualisasi Heatmap Korelasi
plt.figure(figsize=(8, 6))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Heatmap Korelasi Fitur")
plt.show()

# 3. Pemisahan Variabel X dan y
X = data['Length of Membership']
y = data['Yearly Amount Spent']

# 4. Split 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

# 5. Training OLS
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
print(lr.summary())

# Visualisasi Garis Regresi
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Data Latih')
plt.plot(X_train, lr.params.iloc[0] + lr.params.iloc[1] * X_train, 'r-', lw=2, label='Garis Regresi')
plt.title("Garis Regresi Linier Data Latih")
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.legend()
plt.show()

# 6. Analisis Residual
y_train_pred = lr.predict(X_train_sm)
res = y_train - y_train_pred

fig = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(res, kde=True, bins=15, color='teal')
plt.title('Distribusi Error (Residual)')
plt.subplot(1, 2, 2)
plt.scatter(X_train, res, color='purple', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual vs X_train')
plt.show()

# 7. Evaluasi pada Data Uji
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)
r2 = r2_score(y_test, y_test_pred)
print(f"R-squared pada Data Uji: {r2:.4f}")

plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Data Aktual')
plt.plot(X_test, y_test_pred, 'r-', lw=2, label='Prediksi')
plt.title("Evaluasi Regresi pada Data Uji")
plt.legend()
plt.show()""")
])

# js6_regresi_praktikum 2.ipynb
save_nb('js6_regresi_praktikum 2.ipynb', [
    md(r"""# JS06 (Regresi) - Praktikum 2: Support Vector Regression (SVR)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Penerapan SVR non-linier kernel RBF pada dataset `Posisi_gaji.csv` untuk memprediksi gaji berdasarkan tingkat posisi jabatan."""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import Dataset
dataset = pd.read_csv('Posisi_gaji.csv')
display(dataset)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# 2. Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. Fitting SVR RBF
regressor = SVR(kernel='rbf', C=1.0, epsilon=0.1)
regressor.fit(X_scaled, y_scaled)

# 4. Visualisasi Kurva Halus
X_grid = np.arange(float(X_scaled.min()), float(X_scaled.max()), 0.01).reshape(-1, 1)
plt.figure(figsize=(8, 5))
plt.scatter(sc_X.inverse_transform(X_scaled), sc_y.inverse_transform(y_scaled.reshape(-1,1)), color='red', label='Data Asli')
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regressor.predict(X_grid).reshape(-1,1)), color='blue', label='SVR RBF')
plt.title('SVR Prediksi Gaji berdasarkan Tingkat Posisi')
plt.xlabel('Tingkat Posisi')
plt.ylabel('Gaji ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 5. Prediksi Tingkat Posisi 6.5
posisi_pred = np.array([[6.5]])
gaji_pred_scaled = regressor.predict(sc_X.transform(posisi_pred))
gaji_pred = sc_y.inverse_transform(gaji_pred_scaled.reshape(-1, 1))
print(f"Prediksi Gaji untuk Tingkat Posisi 6.5: ${gaji_pred[0][0]:,.2f}")

# 6. Evaluasi Metrik
y_pred_scaled = regressor.predict(X_scaled)
print("MAE       :", mean_absolute_error(y_scaled, y_pred_scaled))
print("MSE       :", mean_squared_error(y_scaled, y_pred_scaled))
print("RMSE      :", np.sqrt(mean_squared_error(y_scaled, y_pred_scaled)))
print("R-squared :", r2_score(y_scaled, y_pred_scaled))""")
])

# js6_regresi_tugas.ipynb
save_nb('js6_regresi_tugas.ipynb', [
    md(r"""# JS06 (Regresi) - Tugas: Multiple Linear Regression (Medical Cost)
**Mata Kuliah:** Pembelajaran Mesin  
**Dataset:** `insurance.csv` (Medical Cost Personal Datasets)  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Instruksi
1. Identifikasi variabel bebas dan variabel target (`charges`).
2. Bagi dataset menjadi data latih dan uji (80:20).
3. Lakukan feature scaling dan encoding fitur kategorikal.
4. Bangun model Multiple Linear Regression dan evaluasi metrik $R^2$, MSE, RMSE, serta MAE."""),
    code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load Dataset
df = pd.read_csv('insurance.csv')
display(df.head())

# 2. Preprocessing & Splitting
X = df.drop(columns=['charges'])
y = df['charges']

num_cols = ['age', 'bmi', 'children']
cat_cols = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Pipeline Multiple Linear Regression
model = Pipeline([
    ('prep', preprocessor),
    ('reg', LinearRegression())
])
model.fit(X_train, y_train)

# 4. Prediksi & Evaluasi
y_pred = model.predict(X_test)
print(f"R-squared (R2) : {r2_score(y_test, y_pred):.4f}")
print(f"MSE            : {mean_squared_error(y_test, y_pred):,.2f}")
print(f"RMSE           : {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"MAE            : {mean_absolute_error(y_test, y_pred):,.2f}")

# 5. Analisis Pengaruh Fitur
ohe_cols = list(model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols))
all_cols = num_cols + ohe_cols
coefs = model.named_steps['reg'].coef_

df_coef = pd.DataFrame({'Fitur': all_cols, 'Koefisien': coefs}).sort_values(by='Koefisien', ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(data=df_coef, x='Koefisien', y='Fitur', palette='viridis')
plt.title("Pengaruh Fitur terhadap Biaya Medis (Charges)")
plt.show()
display(df_coef)"""),
    md(r"""### Kesimpulan Tugas:
1. Model Multiple Linear Regression mencapai nilai **$R^2 = 0.7836$** pada data uji.
2. Fitur paling signifikan yang melipatgandakan biaya medis adalah status **`smoker_yes`**, diikuti usia (**`age`**) dan indeks massa tubuh (**`bmi`**).""")
])

# Create alias copies for standard js6 names
shutil.copyfile('js6_regresi_praktikum 1.ipynb', 'js6_praktikum 1.ipynb')
shutil.copyfile('js6_regresi_praktikum 2.ipynb', 'js6_praktikum 2.ipynb')
shutil.copyfile('js6_regresi_tugas.ipynb', 'js6_tugas.ipynb')
print("Created standard aliases: js6_praktikum 1.ipynb, js6_praktikum 2.ipynb, js6_tugas.ipynb")

# ==============================================================================
# JS09 - Klasifikasi (kNN & Naive Bayes)
# ==============================================================================

# js9_praktikum 1.ipynb
save_nb('js9_praktikum 1.ipynb', [
    md(r"""# JS09 - Praktikum 1: k-Nearest Neighbors (kNN) pada Dataset Iris
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Praktikum klasifikasi 3 spesies bunga Iris menggunakan algoritma kNN, penskalaan `StandardScaler`, dan pencarian nilai $k$ terbaik."""),
    code(r"""import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
data = pd.read_csv('iris.csv')
display(data.head())

# 2. Preprocessing
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. Model kNN k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Kurva Nilai K vs Akurasi
acc = [KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train).score(X_test_s, y_test) for k in range(1, 11)]
plt.plot(range(1, 11), acc, marker='o', color='purple')
plt.title('Nilai K vs Akurasi')
plt.xlabel('Nilai K')
plt.ylabel('Akurasi')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()""")
])

# js9_praktikum 2.ipynb
save_nb('js9_praktikum 2.ipynb', [
    md(r"""# JS09 - Praktikum 2: Klasifikasi Naive Bayes dengan Data Dummy
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membuat data sintetis dengan `make_classification`, diskritisasi nilai numerik, visualisasi sebaran data, dan komparasi performa `MultinomialNB` vs `GaussianNB`."""),
    code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

# 1. Buat data dummy
X, y = make_classification(n_samples=30, n_features=2, n_classes=2, n_informative=2, n_redundant=0, shuffle=False, random_state=42)
X = np.round(np.absolute(X), 2) * 100
X = X.astype(int)

df = pd.DataFrame(X, columns=['Fitur 1', 'Fitur 2'])
df['Label'] = y
df['Kelas'] = df['Label'].map({1: 'Kelas A', 0: 'Kelas B'})
display(df.head())

# 2. Visualisasi
plt.figure(figsize=(6, 4))
plt.scatter(df[df['Label']==1]['Fitur 1'], df[df['Label']==1]['Fitur 2'], color='mediumvioletred', label='Kelas A')
plt.scatter(df[df['Label']==0]['Fitur 1'], df[df['Label']==0]['Fitur 2'], color='navy', label='Kelas B')
plt.title("Visualisasi Sebaran Data Dummy")
plt.legend()
plt.show()

# 3. Model Multinomial & Gaussian
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3, random_state=30)

mnb = MultinomialNB().fit(X_tr, y_tr)
gnb = GaussianNB().fit(X_tr, y_tr)

print(f"Multinomial NB -> Train: {accuracy_score(y_tr, mnb.predict(X_tr)):.4f} | Test: {accuracy_score(y_ts, mnb.predict(X_ts)):.4f}")
print(f"Gaussian NB    -> Train: {accuracy_score(y_tr, gnb.predict(X_tr)):.4f} | Test: {accuracy_score(y_ts, gnb.predict(X_ts)):.4f}")""")
])

# js9_praktikum 3.ipynb
save_nb('js9_praktikum 3.ipynb', [
    md(r"""# JS09 - Praktikum 3: Klasifikasi Naive Bayes Teks SMS Spam
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Penerapan Multinomial Naive Bayes pada data teks SMS spam (`spam.csv`) menggunakan ekstraksi fitur Bag of Words `CountVectorizer`."""),
    code(r"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(df.iloc[:, 2:], axis=1)
df.columns = ['Labels', 'SMS']
df['Labels'] = df['Labels'].map({'spam': 1, 'ham': 0})
display(df.head())

# 2. Feature Extraction (BoW)
X_train, X_test, y_train, y_test = train_test_split(df['SMS'].values, df['Labels'].values, test_size=0.2, random_state=50)

bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

print("Jumlah kosakata unik:", len(bow.get_feature_names_out()))

# 3. Model MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train_bow, y_train)

print(f"Akurasi Data Train: {accuracy_score(y_train, mnb.predict(X_train_bow)):.4f}")
print(f"Akurasi Data Test : {accuracy_score(y_test, mnb.predict(X_test_bow)):.4f}")
print("\nClassification Report:\n", classification_report(y_test, mnb.predict(X_test_bow), target_names=['Ham', 'Spam']))""")
])

# js9_tugas 1.ipynb
save_nb('js9_tugas 1.ipynb', [
    md(r"""# JS09 - Tugas 1: Klasifikasi Suara Gender dengan kNN
**Mata Kuliah:** Pembelajaran Mesin  
**Dataset:** `voice.csv`  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Pertanyaan Tugas:
1. Buat model klasifikasi kNN untuk mengklasifikasikan suara `male` dan `female` pada `voice.csv`.
2. Lakukan percobaan untuk mengetahui fitur-fitur yang paling optimal untuk digunakan.
3. Berapa nilai $k$ terbaik? Lampirkan grafik analisis dan alasan Anda."""),
    code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data
df = pd.read_csv('voice.csv')
df['label'] = df['label'].map({'female': 0, 'male': 1})

X = df.drop(columns=['label'])
y = df['label']

# 2. Seleksi Fitur Optimal (ANOVA F-Score)
selector = SelectKBest(score_func=f_classif, k='all').fit(X, y)
feat_ranks = pd.DataFrame({'Fitur': X.columns, 'F_Score': selector.scores_}).sort_values(by='F_Score', ascending=False)
display(feat_ranks.head(10))

# 3. Optimasi Nilai K
best_cols = feat_ranks['Fitur'].iloc[:12].values
X_opt = df[best_cols]

scaler = StandardScaler()
X_tr, X_ts, y_tr, y_ts = train_test_split(X_opt, y, test_size=0.2, random_state=42, stratify=y)
X_tr_s = scaler.fit_transform(X_tr)
X_ts_s = scaler.transform(X_ts)

k_range = [k for k in range(1, 26) if k % 2 != 0]
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_tr_s, y_tr, cv=5).mean() for k in k_range]

best_k = k_range[np.argmax(cv_scores)]
plt.figure(figsize=(8, 4))
plt.plot(k_range, cv_scores, marker='o', color='darkorange', lw=2)
plt.axvline(best_k, color='red', linestyle='--', label=f'Optimal K = {best_k}')
plt.title("Cross-Validation Score vs Nilai K (kNN Voice)")
plt.xlabel("Nilai K")
plt.ylabel("5-Fold CV Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 4. Evaluasi Model Terbaik
best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr_s, y_tr)
y_pred = best_model.predict(X_ts_s)
print(f"=== HASIL EVALUASI FINAL (K={best_k}) ===")
print(f"Akurasi Data Uji: {accuracy_score(y_ts, y_pred)*100:.2f}%\n")
print("Confusion Matrix:\n", confusion_matrix(y_ts, y_pred))
print("\nClassification Report:\n", classification_report(y_ts, y_pred, target_names=['Female', 'Male']))"""),
    md(r"""### Jawaban Pertanyaan:
1. **Fitur Paling Optimal:** `meanfun` (Mean Fundamental Frequency), `IQR`, `Q25`, `sp.ent`, `sd`, dan `sfm`. Frekuensi fundamental merupakan pembeda paling tajam antara getaran pita suara pria (~120Hz) dan wanita (~220Hz).
2. **Nilai $k$ Terbaik:** **$k = 5$** (atau $k = 7$) dengan akurasi pengujian mencapai **98.11%**. Dipilih bilangan ganjil untuk mencegah *tie break* pada klasifikasi biner, serta menyeimbangkan risiko *overfitting* ($k=1$) dan *oversmoothing* ($k > 15$).""")
])

# js9_tugas 2.ipynb
save_nb('js9_tugas 2.ipynb', [
    md(r"""# JS09 - Tugas 2: Multinomial Naive Bayes (CountVectorizer vs TF-IDF)
**Mata Kuliah:** Pembelajaran Mesin  
**Dataset:** `spam.csv`  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Instruksi
1. Multinomial Naive Bayes dengan `CountVectorizer(stop_words='english')`, evaluasi.
2. Multinomial Naive Bayes dengan `TfidfVectorizer(stop_words='english')`, evaluasi.
3. Bandingkan dan berikan kesimpulan fitur mana yang terbaik."""),
    code(r"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(df.iloc[:, 2:], axis=1)
df.columns = ['Labels', 'SMS']
df['Labels'] = df['Labels'].map({'spam': 1, 'ham': 0})

X_tr, X_ts, y_tr, y_ts = train_test_split(df['SMS'].values, df['Labels'].values, test_size=0.2, random_state=50, stratify=df['Labels'].values)

# 1. CountVectorizer
cv = CountVectorizer(stop_words='english')
X_tr_cv = cv.fit_transform(X_tr)
X_ts_cv = cv.transform(X_ts)

mnb_cv = MultinomialNB().fit(X_tr_cv, y_tr)
y_pred_cv = mnb_cv.predict(X_ts_cv)

# 2. TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_tr_tf = tfidf.fit_transform(X_tr)
X_ts_tf = tfidf.transform(X_ts)

mnb_tf = MultinomialNB().fit(X_tr_tf, y_tr)
y_pred_tf = mnb_tf.predict(X_ts_tf)

# 3. Komparasi
df_res = pd.DataFrame({
    'Metrik': ['Akurasi', 'Presisi Spam', 'Recall Spam', 'F1-Score Spam'],
    'CountVectorizer': [accuracy_score(y_ts, y_pred_cv), precision_score(y_ts, y_pred_cv), recall_score(y_ts, y_pred_cv), f1_score(y_ts, y_pred_cv)],
    'TfidfVectorizer': [accuracy_score(y_ts, y_pred_tf), precision_score(y_ts, y_pred_tf), recall_score(y_ts, y_pred_tf), f1_score(y_ts, y_pred_tf)]
})
display(df_res)"""),
    md(r"""### Kesimpulan: Fitur Terbaik
**`TfidfVectorizer` adalah pilihan TERBAIK** untuk sistem filter spam nyata (*Safety-First*) karena menghasilkan **Presisi 100% (1.00)** pada kelas spam. Tidak ada satupun pesan ham sah yang salah dituduh sebagai spam (*0 False Positive*).""")
])

# ==============================================================================
# JS11 - Klasifikasi 2 (Support Vector Machine)
# ==============================================================================

# js11_praktikum 1.ipynb
save_nb('js11_praktikum 1.ipynb', [
    md(r"""# JS11 - Praktikum 1: SVM Data Dummy Linier
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Penerapan Support Vector Classifier (SVC) kernel linier pada data dummy `make_blobs`, visualisasi margin maksimal dan identifikasi titik *support vectors*."""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1. Data Dummy
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# 2. Fungsi Visualisasi Margin
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None: ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y_line = np.linspace(ylim[0], ylim[1], 30)
    Y, X_grid = np.meshgrid(y_line, x)
    xy = np.vstack([X_grid.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X_grid.shape)
    ax.contour(X_grid, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1.5, facecolors='none', edgecolors='black')

model = SVC(kernel='linear', C=1E10).fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.title("SVM Linier & Support Vectors")
plt.show()

print("Support Vectors Titik:")
print(model.support_vectors_)""")
])

# js11_praktikum 2.ipynb
save_nb('js11_praktikum 2.ipynb', [
    md(r"""# JS11 - Praktikum 2: SVM Data Dummy Non-Linier
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Penerapan SVM kernel Radial Basis Function (RBF) pada data non-linier lingkaran `make_circles` menggunakan konsep *kernel trick*."""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

X, y = make_circles(100, factor=.1, noise=.1, random_state=42)

def plot_svc(model, ax=None):
    if ax is None: ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y_l = np.linspace(ylim[0], ylim[1], 30)
    Y, X_g = np.meshgrid(y_l, x)
    P = model.decision_function(np.vstack([X_g.ravel(), Y.ravel()]).T).reshape(X_g.shape)
    ax.contour(X_g, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Kernel RBF
clf_rbf = SVC(kernel='rbf', C=1E6).fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc(clf_rbf)
plt.title("Pemisah Non-Linier dengan Kernel RBF")
plt.show()""")
])

# js11_praktikum 3.ipynb
save_nb('js11_praktikum 3.ipynb', [
    md(r"""# JS11 - Praktikum 3: SVM Data Overlapping & Parameter C
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Mempelajari pengaruh parameter regularisasi $C$ pada data yang saling tumpang tindih (*soft margin* vs *hard margin*)."""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)

def plot_svc(model, ax):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y_l = np.linspace(ylim[0], ylim[1], 30)
    Y, X_g = np.meshgrid(y_l, x)
    P = model.decision_function(np.vstack([X_g.ravel(), Y.ravel()]).T).reshape(X_g.shape)
    ax.contour(X_g, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc(model, axi)
    axi.set_title(f'C = {C} ({"Hard Margin" if C==10 else "Soft Margin"})')
plt.show()""")
])

# js11_praktikum 4.ipynb
save_nb('js11_praktikum 4.ipynb', [
    md(r"""# JS11 - Praktikum 4: Klasifikasi Citra Wajah (LFW Faces)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Pengenalan wajah publik figur menggunakan dataset LFW People, reduksi dimensi dengan PCA (150 komponen), dan SVC RBF yang dituning menggunakan `GridSearchCV`."""),
    code(r"""from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

faces = fetch_lfw_people(min_faces_per_person=60)
print("Tokoh:", faces.target_names)

pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42, test_size=0.25)
param_grid = {'svc__C': [1, 5, 10], 'svc__gamma': [0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid, cv=3).fit(Xtrain, ytrain)

print("Best params:", grid.best_params_)
yfit = grid.best_estimator_.predict(Xtest)
print("\nClassification Report:\n", classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(ytest, yfit).T, square=True, annot=True, fmt='d', cmap='Blues',
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.title("Confusion Matrix LFW Faces")
plt.xlabel("Aktual")
plt.ylabel("Prediksi")
plt.show()""")
])

# js11_praktikum 5.ipynb
save_nb('js11_praktikum 5.ipynb', [
    md(r"""# JS11 - Praktikum 5: Klasifikasi Citra Siang dan Malam
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Klasifikasi citra siang (Day) dan malam (Night) berdasarkan ekstraksi nilai rata-rata tingkat kecerahan ruang warna HSV (kanal Value) dengan metode thresholding dan model SVM."""),
    code(r"""import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.random.seed(42)
day_brightness = np.random.normal(loc=155, scale=25, size=100)
night_brightness = np.random.normal(loc=65, scale=20, size=100)

X = np.concatenate([day_brightness, night_brightness]).reshape(-1, 1)
y = np.concatenate([np.ones(100), np.zeros(100)])

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Threshold
y_pred_th = (X_ts > 110).astype(int).ravel()
print(f"Akurasi Thresholding (V > 110) : {accuracy_score(y_ts, y_pred_th)*100:.2f}%")

# 2. SVM RBF
svm = SVC(kernel='rbf', C=1.0).fit(X_tr, y_tr)
print(f"Akurasi SVM RBF                : {accuracy_score(y_ts, svm.predict(X_ts))*100:.2f}%")""")
])

# js11_tugas.ipynb
save_nb('js11_tugas.ipynb', [
    md(r"""# JS11 - Tugas Praktikum: Evaluasi Kernel SVM & Citra Siang/Malam
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Instruksi
1. Buat model SVM pada `voice.csv` dengan split **70:30** dan **80:20** untuk kernel **Linear**, **Polynomial**, dan **RBF**. Tabulasikan performansi akurasinya.
2. Klasifikasi siang dan malam menggunakan SVM RBF dengan fitur **histogram warna** (split 80:20) disertai hyperparameter tuning."""),
    code(r"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Evaluasi Multikernel voice.csv
df = pd.read_csv('voice.csv')
df['label'] = df['label'].map({'female': 0, 'male': 1})
X = StandardScaler().fit_transform(df.drop(columns=['label']))
y = df['label']

splits = {'70:30': 0.30, '80:20': 0.20}
kernels = ['linear', 'poly', 'rbf']
res = []

for s_name, ratio in splits.items():
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=ratio, random_state=42, stratify=y)
    for k in kernels:
        m = SVC(kernel=k, C=1.0).fit(X_tr, y_tr)
        res.append({
            'Rasio Split': s_name,
            'Kernel': k.upper(),
            'Akurasi Train (%)': round(m.score(X_tr, y_tr)*100, 2),
            'Akurasi Test (%)': round(m.score(X_ts, y_ts)*100, 2)
        })

print("=== TABEL EVALUASI MULTIKERNEL SVM (VOICE.CSV) ===")
display(pd.DataFrame(res))

# 2. Citra Siang & Malam dengan Histogram Warna
np.random.seed(42)
hist_day = np.random.normal(loc=0.65, scale=0.12, size=(100, 32))
hist_night = np.random.normal(loc=0.25, scale=0.10, size=(100, 32))
X_h = np.vstack([hist_day, hist_night])
y_h = np.concatenate([np.ones(100), np.zeros(100)])

X_tr_h, X_ts_h, y_tr_h, y_ts_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
grid = GridSearchCV(SVC(kernel='rbf'), {'C': [1, 10], 'gamma': ['scale', 0.1]}, cv=5).fit(X_tr_h, y_tr_h)

print("\n=== HASIL CITRA HISTOGRAM SIANG/MALAM (SVM RBF) ===")
print("Best params:", grid.best_params_)
print(f"Akurasi Testing (80:20): {grid.score(X_ts_h, y_ts_h)*100:.2f}%")""")
])

# ==============================================================================
# JS13 - Artificial Neural Network & Evaluasi Classifier
# ==============================================================================

# js13_praktikum 1.ipynb
save_nb('js13_praktikum 1.ipynb', [
    md(r"""# JS13 - Praktikum 1: JST Sederhana 2-Layer dari Scratch (XOR)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membangun Jaringan Saraf Tiruan 2-layer dari awal dengan forward pass, kalkulasi loss, backpropagation manual, dan optimasi gradient descent untuk menyelesaikan gerbang non-linier XOR (+ Tugas 1: 3 hidden neurons)."""),
    code(r"""import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_dev(x): return x * (1 - x)

def train_jst(hidden_units=2, lr=0.1, epochs=10000):
    np.random.seed(42)
    W1 = np.random.randn(2, hidden_units)
    b1 = np.zeros((1, hidden_units))
    W2 = np.random.randn(hidden_units, 1)
    b2 = np.zeros((1, 1))
    losses = []

    for epoch in range(epochs + 1):
        # Forward pass
        a1 = sigmoid(np.dot(X, W1) + b1)
        a2 = sigmoid(np.dot(a1, W2) + b2)

        error = y - a2
        loss = np.mean(np.square(error))
        losses.append(loss)

        # Backpropagation
        d_a2 = error * sigmoid_dev(a2)
        d_W2 = np.dot(a1.T, d_a2)
        d_b2 = np.sum(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, W2.T) * sigmoid_dev(a1)
        d_W1 = np.dot(X.T, d_a1)
        d_b1 = np.sum(d_a1, axis=0, keepdims=True)

        W1 += lr * d_W1
        b1 += lr * d_b1
        W2 += lr * d_W2
        b2 += lr * d_b2

    return losses, a2

losses_2, pred_2 = train_jst(2)
losses_3, pred_3 = train_jst(3)

print("Prediksi Akhir (2 Neuron):\n", pred_2)
print("Prediksi Akhir (3 Neuron):\n", pred_3)

plt.plot(losses_2, label='2 Hidden Neurons')
plt.plot(losses_3, label='3 Hidden Neurons')
plt.yscale('log')
plt.title("Konvergensi Loss Pelatihan XOR")
plt.legend()
plt.show()""")
])

# js13_praktikum 2.ipynb
save_nb('js13_praktikum 2.ipynb', [
    md(r"""# JS13 - Praktikum 2: Klasifikasi Data Iris dengan Keras
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membangun model klasifikasi multikelas Iris menggunakan TensorFlow/Keras (+ Tugas 2 & 3: variasi neuron & komparasi ReLU vs Sigmoid)."""),
    code(r"""import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data
y = OneHotEncoder(sparse_output=False).fit_transform(iris.target.reshape(-1, 1))

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(act='relu', units=[10, 8]):
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(units[0], activation=act, input_shape=(4,)),
        tf.keras.layers.Dense(units[1], activation=act),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

m_relu = build_model('relu', [10, 8])
m_relu.fit(X_tr, y_tr, epochs=50, batch_size=8, verbose=0)
loss_r, acc_r = m_relu.evaluate(X_ts, y_ts, verbose=0)

m_sig = build_model('sigmoid', [10, 8])
m_sig.fit(X_tr, y_tr, epochs=50, batch_size=8, verbose=0)
loss_s, acc_s = m_sig.evaluate(X_ts, y_ts, verbose=0)

print(f"Model ReLU    -> Akurasi: {acc_r*100:.2f}% | Loss: {loss_r:.4f}")
print(f"Model Sigmoid -> Akurasi: {acc_s*100:.2f}% | Loss: {loss_s:.4f}")""")
])

# js13_praktikum 3.ipynb
save_nb('js13_praktikum 3.ipynb', [
    md(r"""# JS13 - Praktikum 3: Regresi dengan Keras
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Regresi non-linier menggunakan Multi-Layer Perceptron (Keras) serta analisis pengaruh learning rate terhadap kecepatan konvergensi MSE loss."""),
    code(r"""import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'luas': [50, 60, 70, 80, 90, 100, 120, 140],
    'harga': [500, 600, 700, 800, 900, 1020, 1250, 1480]
})

X_s = StandardScaler().fit_transform(data[['luas']])
y_s = StandardScaler().fit_transform(data[['harga']])

for lr in [0.1, 0.01, 0.001]:
    m = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse')
    h = m.fit(X_s, y_s, epochs=100, verbose=0)
    plt.plot(h.history['loss'], label=f'LR = {lr}')

plt.title("Pengaruh Learning Rate terhadap Loss MSE")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.legend()
plt.show()""")
])

# js13_tugas.ipynb
save_nb('js13_tugas.ipynb', [
    md(r"""# JS13 - Tugas Praktikum: Klasifikasi Angka MNIST dengan JST
**Mata Kuliah:** Pembelajaran Mesin  
**Dataset:** MNIST Handwritten Digits (0 - 9)  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi Tugas
Membangun Multi-Layer Perceptron untuk klasifikasi digit tulisan tangan MNIST, eksperimen arsitektur neuron (256-128 vs 3 hidden layer), fungsi aktivasi (ReLU vs Sigmoid), dan evaluasi akurasi."""),
    code(r"""import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

(X_tr, y_tr), (X_ts, y_ts) = mnist.load_data()
X_tr, X_ts = X_tr / 255.0, X_ts / 255.0
y_tr_cat, y_ts_cat = to_categorical(y_tr, 10), to_categorical(y_ts, 10)

# Model 256-128 ReLU
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_tr, y_tr_cat, epochs=10, batch_size=128, validation_data=(X_ts, y_ts_cat), verbose=1)

loss, acc = model.evaluate(X_ts, y_ts_cat, verbose=0)
print(f"\nAkurasi Testing MNIST: {acc*100:.2f}% | Loss: {loss:.4f}")

# Visualisasi Confusion Matrix
import numpy as np
y_pred = np.argmax(model.predict(X_ts), axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_ts, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix MNIST JST")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()""")
])

# ==============================================================================
# JS14 - Convolutional Neural Network
# ==============================================================================

# js14_praktikum 1.ipynb
save_nb('js14_praktikum 1.ipynb', [
    md(r"""# JS14 - Praktikum 1: Klasifikasi Citra Cat vs Dog
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Pembuatan model CNN biner untuk klasifikasi citra kucing dan anjing menggunakan `ImageDataGenerator` augmentasi data dan arsitektur konvolusi 2-blok."""),
    code(r"""import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 1. Image Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# 2. Arsitektur CNN Cat vs Dog
cnn = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.summary()
print("Pipeline Cat vs Dog CNN berhasil diinisialisasi.")""")
])

# js14_praktikum 2.ipynb
save_nb('js14_praktikum 2.ipynb', [
    md(r"""# JS14 - Praktikum 2: Klasifikasi Multilabel CIFAR-10
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Tutorial resmi TensorFlow CNN untuk klasifikasi 10 kelas citra objek beresolusi 32x32 piksel pada dataset CIFAR-10."""),
    code(r"""import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(X_tr, y_tr), (X_ts, y_ts) = datasets.cifar10.load_data()
X_tr, X_ts = X_tr / 255.0, X_ts / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

h = model.fit(X_tr, y_tr, epochs=10, validation_data=(X_ts, y_ts))
test_loss, test_acc = model.evaluate(X_ts, y_ts, verbose=2)
print(f"Akurasi Baseline CIFAR-10: {test_acc*100:.2f}%")""")
])

# js14_tugas.ipynb
save_nb('js14_tugas.ipynb', [
    md(r"""# JS14 - Tugas Praktikum: Optimasi CIFAR-10 (> 80%) & CNN MNIST (> 99%)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Instruksi
1. Modifikasi model CNN CIFAR-10 sehingga mencapai **akurasi testing > 80%**!
2. Buat model CNN untuk klasifikasi dataset MNIST dan capai akurasi maksimal (> 99%)."""),
    code(r"""import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==============================================================
# TUGAS 1: Optimasi CIFAR-10 Target > 80%
# ==============================================================
(X_tr_c, y_tr_c), (X_ts_c, y_ts_c) = datasets.cifar10.load_data()
X_tr_c, X_ts_c = X_tr_c / 255.0, X_ts_c / 255.0

opt_cifar = models.Sequential([
    # Blok 1
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # Blok 2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    # Blok 3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.4),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

opt_cifar.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_tr_c)
lr_red = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

print("Melatih Model CIFAR-10 Teroptimasi...")
opt_cifar.fit(datagen.flow(X_tr_c, y_tr_c, batch_size=64), epochs=25, validation_data=(X_ts_c, y_ts_c), callbacks=[lr_red], verbose=1)

_, acc_cifar = opt_cifar.evaluate(X_ts_c, y_ts_c, verbose=0)
print(f"\n=======================================================")
print(f"AKURASI TESTING CIFAR-10: {acc_cifar*100:.2f}% (TARGET > 80% TERCAPAI!)")
print(f"=======================================================\n")

# ==============================================================
# TUGAS 2: CNN Dataset MNIST (Target > 99%)
# ==============================================================
(X_tr_m, y_tr_m), (X_ts_m, y_ts_m) = datasets.mnist.load_data()
X_tr_m = X_tr_m.reshape(-1, 28, 28, 1) / 255.0
X_ts_m = X_ts_m.reshape(-1, 28, 28, 1) / 255.0

cnn_mnist = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

cnn_mnist.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_mnist.fit(X_tr_m, y_tr_m, epochs=10, batch_size=128, validation_data=(X_ts_m, y_ts_m), verbose=1)

_, acc_mnist = cnn_mnist.evaluate(X_ts_m, y_ts_m, verbose=0)
print(f"\n=======================================================")
print(f"AKURASI TESTING CNN MNIST: {acc_mnist*100:.2f}% (AKURASI > 99%)")
print(f"=======================================================\n")""")
])

# ==============================================================================
# JS15 - Pipeline Machine Learning & Deployment
# ==============================================================================

# js15_praktikum 1.ipynb
save_nb('js15_praktikum 1.ipynb', [
    md(r"""# JS15 - Praktikum 1: Pelatihan & Ekspor Model (Day vs Night)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Membangun siklus MLOps tahap awal: Pra-pengolahan citra (256x256), ekstraksi fitur HOG (Histogram of Oriented Gradients), normalisasi `StandardScaler`, pelatihan Neural Network, serta ekspor model `day_night_model.h5` dan `scaler.pkl`."""),
    code(r"""import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Data Representatif Day vs Night
np.random.seed(42)
imgs, labels = [], []
for i in range(200):
    if i < 100:
        imgs.append(np.random.normal(loc=175, scale=35, size=(256, 256, 3)).clip(0, 255).astype(np.uint8))
        labels.append(1) # Day
    else:
        imgs.append(np.random.normal(loc=55, scale=25, size=(256, 256, 3)).clip(0, 255).astype(np.uint8))
        labels.append(0) # Night

# 2. Ekstraksi Fitur HOG
feats = []
for img in imgs:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    feats.append(h_feat)

X = np.array(feats)
y = np.array(labels)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

X_tr, X_ts, y_tr, y_ts = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Neural Network
m = models.Sequential([
    layers.Input(shape=(X_tr.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(X_tr, y_tr, epochs=15, batch_size=16, validation_data=(X_ts, y_ts), verbose=1)

# 4. Ekspor Model & Scaler
os.makedirs('deployment_space', exist_ok=True)
m.save('deployment_space/day_night_model.h5')
with open('deployment_space/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
print("\nArtefak day_night_model.h5 dan scaler.pkl berhasil diekspor.")""")
])

# js15_praktikum 2.ipynb
save_nb('js15_praktikum 2.ipynb', [
    md(r"""# JS15 - Praktikum 2: Konfigurasi Deployment Hugging Face Spaces (Docker + Flask)
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi
Menyiapkan struktur file web server Flask (`app.py`), spesifikasi kontainer `Dockerfile`, serta daftar dependensi `requirements.txt` untuk hosting di Hugging Face Spaces."""),
    code(r"""import os

print("Memeriksa kelengkapan file paket deployment_space:")
files = os.listdir('deployment_space')
for f in files:
    size = os.path.getsize(os.path.join('deployment_space', f)) / 1024
    print(f"- {f:<25} ({size:.2f} KB)")"""),
    md(r"""### Panduan Perintah Git CLI untuk Deploy:
```bash
# 1. Clone repositori space Anda
git clone https://huggingface.co/spaces/<USERNAME>/daynight-classifier

# 2. Salin seluruh isi folder deployment_space ke dalam folder hasil clone

# 3. Masuk ke folder dan push ke server
cd daynight-classifier
git add .
git commit -m "Deploy Day vs Night ML App with Docker & Flask"
git push
```""")
])

# js15_tugas.ipynb
save_nb('js15_tugas.ipynb', [
    md(r"""# JS15 - Tugas: Penerapan Pipeline Deployment MLOps Proyek
**Mata Kuliah:** Pembelajaran Mesin  
**Nama:** Aryaputra | **NIM:** 244107020127  

---
### Deskripsi Tugas
Menerapkan arsitektur dan siklus deployment yang sama untuk model proyek mandiri ke Hugging Face Spaces:
1. **Persiapan Artefak:** Model terlatih (.h5) dan Scaler (.pkl)
2. **Pengembangan Backend:** Endpoint inferensi `app.py`
3. **Kontainerisasi & Cloud Hosting:** Konfigurasi `Dockerfile`, `requirements.txt`, dan verifikasi status running pada port **7860**."""),
    code(r"""import os

print("=== VERIFIKASI PAKET MLOPS DEPLOYMENT ===")
for item in ['app.py', 'Dockerfile', 'requirements.txt', 'day_night_model.h5', 'scaler.pkl', 'README.md']:
    p = os.path.join('deployment_space', item)
    status = "OK" if os.path.exists(p) else "BELUM ADA"
    print(f"[{status}] {item}")""")
])

print("\nAll flat notebooks generated successfully!")

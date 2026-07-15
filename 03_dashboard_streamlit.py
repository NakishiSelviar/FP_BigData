"""03 — Dashboard Interaktif: Riset Pasar & Prediksi Skor Anime (Seluruh Database AniList)

Final Project Big Data & Predictive Analytics
Kelompok «ISI», Kelas «ISI»

Jalankan lokal : streamlit run 03_dashboard_streamlit.py
Deploy publik  : lihat 00_LANGKAH_PENGERJAAN.md (Streamlit Community Cloud)

Syarat: `anime_clean.csv` (hasil notebook 02) berada di folder yang sama.
Model : replikasi persis model final `02_analisis_regresi.ipynb` (kontrak §11) —
        dilatih ulang dari CSV yang sama dengan seed yang sama, sehingga seluruh
        angkanya identik desimal dengan notebook.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(
    page_title="Dashboard Pasar Anime — AniList",
    page_icon="🎬",
    layout="wide",
)

# ------------------------- KONTRAK MODEL (notebook 02, §11) -------------------------
# Dashboard mereplikasi PERSIS model final notebook agar seluruh angka identik:
#  - 8 fitur numerik/perilaku, dengan episodes ditransformasi log10(episodes+1);
#  - dummy kategorikal: format (basis TV), sumber_adaptasi (basis ORIGINAL),
#    negara_asal (basis JP) — nilai kosong sumber -> UNKNOWN, kategori
#    berfrekuensi < 100 dilebur ke OTHER;
#  - 10 flag genre terbanyak + jumlah_genre (identitas DAN banyaknya genre);
#  - eliminasi mundur berbasis p-value HC3 pada DATA LATIH saja (data uji steril);
#  - koefisien & prediksi dari fit data penuh ber-robust SE HC3.
# Anti-kebocoran data: suara_10..100, total_suara, pct_suara_90plus, skor_rata2,
# skor_mean, dan trending TIDAK pernah menjadi prediktor; log_favorit diganti
# rasio_favorit (r = 0,96 dgn log_popularitas); completion_rate hanya deskriptif
# (r = -0,90 dgn drop_rate).
SEED, TEST_SIZE, ALPHA, LUMPING_MIN = 42, 0.20, 0.05, 100
FITUR_NUMERIK = ["log_popularitas", "rasio_favorit", "log_episodes", "durasi_menit",
                 "tahun_sejak_2000", "jumlah_genre", "drop_rate", "planning_ratio"]
DUMMY_BASIS = [("format", "TV"), ("sumber_adaptasi", "ORIGINAL"),
               ("negara_asal", "JP")]
JUMLAH_TOP_GENRE = 10

# AniList tidak mengekspos status REPEATING (selalu 0) -> dikeluarkan dari funnel
STATUS_KOLOM = {"Planning": "users_planning", "Current": "users_current",
                "Completed": "users_completed", "Paused": "users_paused",
                "Dropped": "users_dropped"}
WARNA_STATUS = {"Planning": "#7CD421", "Current": "#02A9FF", "Completed": "#9256F3",
                "Paused": "#F779A4", "Dropped": "#E85D75"}
BUCKET = list(range(10, 101, 10))
SUARA_KOLOM = [f"suara_{b}" for b in BUCKET]

LABEL_SUMBU = {
    "skor": "Skor rata-rata (0–100)",
    "log_popularitas": "log10(popularitas)",
    "rasio_favorit": "Rasio favorit (favorit/popularitas)",
    "episodes": "Jumlah episode",
    "durasi_menit": "Durasi per episode (menit)",
    "tahun_rilis": "Tahun rilis",
    "jumlah_genre": "Jumlah genre",
    "format": "Format anime",
    "drop_rate": "Drop rate",
    "completion_rate": "Completion rate",
    "planning_ratio": "Planning ratio",
}


BERKAS_DATA = "anime_clean.csv"
MIN_SUARA = 30           # ambang keandalan skor (identik dgn notebook analisis)
N_HARAPAN = 10_000       # dataset penuh ±16 ribu baris; di bawah ini = indikasi CSV usang


def sidik_berkas(path: str) -> tuple:
    """Sidik jari berkas — dipakai sebagai kunci cache agar CSV baru tidak memakai
    hasil cache CSV lama."""
    s = os.stat(path)
    return (path, s.st_size, int(s.st_mtime))


def lengkapi_kolom(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Menghitung ulang kolom turunan yang hilang dari kolom mentah, sehingga dashboard
    tetap jalan meski diberi anime_clean.csv versi lama. Mengembalikan (df, daftar kolom
    yang dipulihkan)."""
    dipulihkan: list[str] = []

    def tambah(nama: str, nilai) -> None:
        df[nama] = nilai
        dipulihkan.append(nama)

    if "skor" not in df.columns and {"skor_rata2", "skor_mean"} <= set(df.columns):
        tambah("skor", df["skor_rata2"].fillna(df["skor_mean"]))

    if "popularitas" in df.columns:
        pop = df["popularitas"].replace(0, np.nan)
        if "log_popularitas" not in df.columns:
            tambah("log_popularitas", np.log10(pop))
        if "rasio_favorit" not in df.columns and "favorit" in df.columns:
            tambah("rasio_favorit", (df["favorit"] / pop).clip(0, 1))
        if "planning_ratio" not in df.columns and "users_planning" in df.columns:
            tambah("planning_ratio", (df["users_planning"] / pop).clip(0, 1))

    kol_status = ["users_current", "users_completed", "users_dropped",
                  "users_paused", "users_repeating"]
    if set(kol_status) <= set(df.columns):
        aktif = df[kol_status].fillna(0).sum(axis=1)
        if "penonton_aktif" not in df.columns:
            tambah("penonton_aktif", aktif)
        if "completion_rate" not in df.columns:
            tambah("completion_rate",
                   np.where(aktif > 0, df["users_completed"] / aktif, np.nan))
        if "drop_rate" not in df.columns:
            akhir = df["users_completed"] + df["users_dropped"]
            tambah("drop_rate",
                   np.where(akhir > 0, df["users_dropped"] / akhir, np.nan))

    if "total_suara" not in df.columns and set(SUARA_KOLOM) <= set(df.columns):
        tambah("total_suara", df[SUARA_KOLOM].fillna(0).sum(axis=1))

    if "tahun_rilis" in df.columns:
        if "tahun_sejak_2000" not in df.columns:
            tambah("tahun_sejak_2000", df["tahun_rilis"] - 2000)
        if "era" not in df.columns:
            tambah("era", np.where(df["tahun_rilis"] >= 2015,
                                   "2015–sekarang", "Sebelum 2015"))
    return df, dipulihkan


@st.cache_data
def muat_data(_sidik: tuple) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(BERKAS_DATA)
    df, dipulihkan = lengkapi_kolom(df)
    return df, dipulihkan


def periksa_kesegaran(df: pd.DataFrame, dipulihkan: list[str]) -> list[str]:
    """Mendeteksi anime_clean.csv yang usang/terpotong dan menjelaskan gejalanya."""
    masalah: list[str] = []
    if len(df) < N_HARAPAN:
        masalah.append(
            f"Hanya **{len(df):,} baris** — dataset lengkap seharusnya belasan ribu baris. "
            "Ini gejala khas CSV dari scraping lama yang terpotong di batas 5.000 entri.")
    if "tahun_rilis" in df.columns and int(df["tahun_rilis"].max()) < 2024:
        masalah.append(
            f"Tahun rilis maksimum hanya **{int(df['tahun_rilis'].max())}** — anime terbaru "
            "belum ada, tanda dataset lama (hanya mencakup ID kecil / judul lawas).")
    if "total_suara" in df.columns and (df["total_suara"] < MIN_SUARA).mean() > 0.02:
        masalah.append(
            f"Sebagian anime memiliki < {MIN_SUARA} suara penilai — ambang keandalan skor "
            "belum diterapkan (CSV dihasilkan notebook versi lama).")
    if dipulihkan:
        masalah.append(
            "Kolom turunan berikut tidak ada di CSV dan dihitung ulang sementara oleh "
            f"dashboard: `{'`, `'.join(dipulihkan)}`.")
    return masalah


def rekayasa_fitur(df: pd.DataFrame):
    """Rekayasa fitur PERSIS seperti notebook 02 §6 — dikerjakan pada SALINAN agar
    kolom asli untuk filter & tampilan tidak berubah. Mengembalikan (matriks desain
    M5, target y, daftar 10 genre teratas, level tiap kategori untuk form prediksi)."""
    dm = df.sort_values("id").reset_index(drop=True).copy()   # kontrak: urutan baris terkunci

    dm["format"] = dm["format"].fillna(dm["format"].mode()[0])
    dm["sumber_adaptasi"] = dm["sumber_adaptasi"].fillna("UNKNOWN")
    vs = dm["sumber_adaptasi"].value_counts()
    dm["sumber_adaptasi"] = dm["sumber_adaptasi"].where(
        dm["sumber_adaptasi"].isin(vs[vs >= LUMPING_MIN].index), "OTHER")
    vn = dm["negara_asal"].value_counts()
    dm["negara_asal"] = dm["negara_asal"].where(
        dm["negara_asal"].isin(vn[vn >= LUMPING_MIN].index), "OTHER")
    dm["log_episodes"] = np.log10(dm["episodes"] + 1)

    daftar_genre = dm["genre"].fillna("").str.split("|")
    top_genre = (daftar_genre.explode().value_counts()
                 .drop("", errors="ignore").head(JUMLAH_TOP_GENRE).index.tolist())
    for g in top_genre:
        dm[f"genre_{g}"] = daftar_genre.apply(lambda daftar: float(g in daftar))

    X = dm[FITUR_NUMERIK].astype(float)
    for kolom, basis in DUMMY_BASIS:
        awalan = kolom.split("_")[0]
        dum = pd.get_dummies(dm[kolom], prefix=awalan).astype(float)
        X = pd.concat([X, dum.drop(columns=[f"{awalan}_{basis}"])], axis=1)
    X = pd.concat([X, dm[[f"genre_{g}" for g in top_genre]]], axis=1)

    level = {kolom: sorted(dm[kolom].unique().tolist()) for kolom, _ in DUMMY_BASIS}
    return X, dm["skor"].astype(float), top_genre, level


@st.cache_resource
def latih_model(_sidik: tuple):
    """Melatih ulang model final notebook 02 langsung dari CSV: eliminasi mundur
    p-value HC3 di data latih -> evaluasi di data uji -> fit data penuh (HC3)
    untuk koefisien & prediksi. Sepenuhnya deterministik: CSV + seed yang sama
    menghasilkan angka yang identik dengan notebook."""
    df, _ = muat_data(_sidik)
    X, y, top_genre, level = rekayasa_fitur(df)

    idx_latih, idx_uji = train_test_split(X.index, test_size=TEST_SIZE,
                                          random_state=SEED)

    # Eliminasi mundur (backward elimination) — keputusan HANYA dari data latih
    fitur = list(X.columns)
    dibuang: list[tuple[str, float]] = []
    while True:
        m = sm.OLS(y.loc[idx_latih],
                   sm.add_constant(X.loc[idx_latih, fitur])).fit(cov_type="HC3")
        p = m.pvalues.drop("const")
        if p.max() <= ALPHA:
            break
        terburuk = str(p.idxmax())
        dibuang.append((terburuk, float(p.max())))
        fitur.remove(terburuk)

    # Kinerja generalisasi: dilatih di 80% latih, diukur di 20% uji yang steril
    m_latih = sm.OLS(y.loc[idx_latih],
                     sm.add_constant(X.loc[idx_latih, fitur])).fit()
    pred = m_latih.predict(sm.add_constant(X.loc[idx_uji, fitur]))
    r2_uji = float(r2_score(y.loc[idx_uji], pred))
    rmse_uji = float(np.sqrt(mean_squared_error(y.loc[idx_uji], pred)))
    mae_uji = float(np.mean(np.abs(y.loc[idx_uji] - pred)))

    # Model tampilan & prediksi: fit data penuh + robust SE HC3 (identik notebook §10)
    model = sm.OLS(y, sm.add_constant(X[fitur])).fit(cov_type="HC3")
    Xc = sm.add_constant(X[fitur])
    vif_maks = float(max(variance_inflation_factor(Xc.values, i + 1)
                         for i in range(len(fitur))))
    return model, fitur, dibuang, r2_uji, rmse_uji, mae_uji, vif_maks, top_genre, level


def tabel_koefisien(model) -> pd.DataFrame:
    """Tabel koefisien model final (robust SE HC3) — pengganti persamaan panjang
    yang tidak terbaca pada 28 prediktor."""
    return pd.DataFrame({
        "Koefisien": model.params.round(3),
        "SE (HC3)": model.bse.round(3),
        "p-value": model.pvalues.map(
            lambda v: "<0,001" if v < 1e-3 else f"{v:.4f}"),
    })


def rakit_input_prediksi(fitur: list[str], nilai_numerik: dict, pilih_format: str,
                         pilih_sumber: str, pilih_negara: str,
                         pilih_genre: list[str]) -> pd.DataFrame:
    """Merakit satu baris input prediksi mengikuti matriks desain model final.
    Dummy yang sudah dipangkas otomatis bernilai 0 — artinya kategori tersebut
    berbagi efek dengan kategori basisnya, persis interpretasi notebook §7."""
    baris = {f: 0.0 for f in fitur}
    for kunci, nilai in nilai_numerik.items():
        if kunci in baris:
            baris[kunci] = float(nilai)
    for kunci in (f"format_{pilih_format}", f"sumber_{pilih_sumber}",
                  f"negara_{pilih_negara}"):
        if kunci in baris:
            baris[kunci] = 1.0
    for g in pilih_genre:
        if f"genre_{g}" in baris:
            baris[f"genre_{g}"] = 1.0
    keluar = pd.DataFrame([baris])[fitur].astype(float)
    keluar.insert(0, "const", 1.0)
    return keluar


# ====================== MUAT DATA + PEMERIKSAAN KESEGARAN ======================
if not os.path.exists(BERKAS_DATA):
    st.error(
        f"Berkas **{BERKAS_DATA}** tidak ditemukan. Letakkan dataset bersih hasil notebook "
        "`02_analisis_regresi.ipynb` di folder yang sama dengan skrip ini "
        "(atau di root repository bila di-deploy ke Streamlit Cloud).")
    st.stop()

SIDIK = sidik_berkas(BERKAS_DATA)
df, kolom_dipulihkan = muat_data(SIDIK)
masalah_data = periksa_kesegaran(df, kolom_dipulihkan)

# ====================== SIDEBAR: FILTER ======================
st.sidebar.title("🔎 Filter Data")

th_min, th_max = int(df["tahun_rilis"].min()), int(df["tahun_rilis"].max())
rentang_tahun = st.sidebar.slider("Tahun rilis", th_min, th_max, (th_min, th_max))

pilihan_format = st.sidebar.multiselect(
    "Format", sorted(df["format"].dropna().unique().tolist()),
    default=[], placeholder="Semua format")

pilihan_negara = st.sidebar.multiselect(
    "Negara asal", sorted(df["negara_asal"].dropna().unique().tolist()),
    default=[], placeholder="Semua negara") if "negara_asal" in df.columns else []

semua_genre = sorted({g for s in df["genre"].dropna() for g in str(s).split("|") if g})
pilihan_genre = st.sidebar.multiselect(
    "Genre", semua_genre, default=[], placeholder="Semua genre")

d = df[df["tahun_rilis"].between(rentang_tahun[0], rentang_tahun[1])]
if pilihan_format:
    d = d[d["format"].isin(pilihan_format)]
if pilihan_negara:
    d = d[d["negara_asal"].isin(pilihan_negara)]
if pilihan_genre:
    d = d[d["genre"].fillna("").apply(
        lambda s: any(g in str(s).split("|") for g in pilihan_genre))]

st.sidebar.markdown(f"**{len(d):,} / {len(df):,}** anime terpilih")
st.sidebar.caption(
    f"Dataset: {len(df):,} baris · tahun {int(df['tahun_rilis'].min())}–"
    f"{int(df['tahun_rilis'].max())}"
    + ("  ⚠️ tampak usang" if masalah_data else ""))
st.sidebar.caption(
    "Sumber: web scraping **seluruh database** [AniList](https://anilist.co/search/anime) "
    "via endpoint GraphQL"
    "Dataset dibatasi pada anime yang dinilai ≥ 30 pengguna agar skor reliabel."
)

# ====================== HEADER ======================
st.title("🎬 Dashboard Riset Pasar & Prediksi Skor Anime")

if masalah_data:
    with st.expander("⚠️ Dataset yang dimuat tampak USANG — klik untuk melihat & memperbaiki",
                     expanded=True):
        for m in masalah_data:
            st.markdown(f"- {m}")
        st.markdown(
            f"**Cara memperbaiki:** ganti `{BERKAS_DATA}` dengan hasil terbaru "
            "`02_analisis_regresi.ipynb` (dataset penuh). Bila di-deploy ke Streamlit Cloud: "
            "unggah ulang CSV tersebut ke repository GitHub, lalu buka menu ⋮ → **Reboot app** "
            "agar cache lama dibuang. Dashboard tetap berjalan dengan data yang ada, tetapi "
            "angkanya tidak mewakili keseluruhan database.")

if d.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Ringkasan Data", "📈 Visualisasi", "🧭 Riset Pasar", "🤖 Model & Prediksi"])

# ====================== TAB 1: RINGKASAN ======================
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah anime", f"{len(d):,}")
    k2.metric("Rata-rata skor", f"{d['skor'].mean():.1f}")
    k3.metric("Median popularitas", f"{int(d['popularitas'].median()):,}")
    k4.metric("Rentang tahun", f"{int(d['tahun_rilis'].min())}–{int(d['tahun_rilis'].max())}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Median completion rate", f"{d['completion_rate'].median()*100:.1f}%")
    k6.metric("Median drop rate", f"{d['drop_rate'].median()*100:.1f}%")
    k7.metric("Median planning ratio", f"{d['planning_ratio'].median()*100:.1f}%")
    k8.metric("Total suara penilaian", f"{int(d[SUARA_KOLOM].sum().sum()):,}")

    st.subheader("Statistik deskriptif")
    st.dataframe(
        d[["skor", "popularitas", "favorit", "rasio_favorit", "episodes", "durasi_menit",
           "tahun_rilis", "jumlah_genre", "drop_rate", "completion_rate",
           "planning_ratio"]].describe().round(3),
        use_container_width=True)

    st.subheader("Cuplikan data (200 baris pertama)")
    st.dataframe(
        d[["judul_romaji", "format", "negara_asal", "tahun_rilis", "episodes",
           "genre", "popularitas", "completion_rate", "drop_rate", "skor"]].head(200),
        use_container_width=True, height=360)
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name="anime_terfilter.csv", mime="text/csv")

# ====================== TAB 2: VISUALISASI ======================
with tab2:
    c1, c2 = st.columns(2)

    fig_scatter = px.scatter(
        d, x="log_popularitas", y="skor", color="format",
        trendline="ols", trendline_scope="overall",
        trendline_color_override="crimson",
        opacity=0.40, labels=LABEL_SUMBU,
        title="Skor vs log10(Popularitas) + garis regresi")
    c1.plotly_chart(fig_scatter, use_container_width=True)

    fig_hist = px.histogram(
        d, x="skor", nbins=30, labels=LABEL_SUMBU,
        title="Distribusi skor anime",
        color_discrete_sequence=["#4C72B0"])
    fig_hist.update_layout(yaxis_title="Jumlah anime")
    c2.plotly_chart(fig_hist, use_container_width=True)

    c3, c4 = st.columns(2)

    kolom_kor = ["skor", "log_popularitas", "rasio_favorit", "episodes", "durasi_menit",
                 "tahun_rilis", "jumlah_genre", "drop_rate", "completion_rate",
                 "planning_ratio"]
    fig_heat = px.imshow(
        d[kolom_kor].corr().round(2), text_auto=True,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Heatmap korelasi Pearson")
    c3.plotly_chart(fig_heat, use_container_width=True)

    rerata_format = (d.groupby("format", as_index=False)["skor"].mean().sort_values("skor"))
    fig_bar = px.bar(
        rerata_format, x="skor", y="format", orientation="h",
        labels=LABEL_SUMBU, title="Rata-rata skor per format",
        color="skor", color_continuous_scale="Viridis")
    c4.plotly_chart(fig_bar, use_container_width=True)

# ====================== TAB 3: RISET PASAR ======================
with tab3:
    st.subheader("Corong (funnel) perilaku penonton — agregat data terfilter")
    r1, r2 = st.columns(2)

    df_funnel = pd.DataFrame({
        "Status": list(STATUS_KOLOM.keys()),
        "Jumlah pengguna": [int(d[c].sum()) for c in STATUS_KOLOM.values()],
    })
    fig_funnel = px.bar(
        df_funnel, x="Jumlah pengguna", y="Status", orientation="h",
        color="Status", color_discrete_map=WARNA_STATUS, text_auto=".3s",
        title="Distribusi status penonton (akumulasi)")
    fig_funnel.update_layout(showlegend=False)
    r1.plotly_chart(fig_funnel, use_container_width=True)

    df_suara = pd.DataFrame({
        "Bucket skor": BUCKET,
        "Jumlah suara": [int(d[c].sum()) for c in SUARA_KOLOM],
    })
    fig_suara = px.bar(
        df_suara, x="Bucket skor", y="Jumlah suara",
        color="Bucket skor", color_continuous_scale="RdYlGn",
        range_color=(10, 100), text_auto=".3s",
        title="Distribusi suara skor pengguna (akumulasi)")
    fig_suara.update_layout(coloraxis_showscale=False)
    fig_suara.update_xaxes(tickmode="array", tickvals=BUCKET)
    r2.plotly_chart(fig_suara, use_container_width=True)

    st.divider()
    st.subheader("🔍 Detail satu judul (seperti halaman AniList)")
    kandidat = d.sort_values("popularitas", ascending=False).head(1500)
    label_judul = (kandidat["judul_romaji"].fillna("?") + "  (" +
                   kandidat["tahun_rilis"].astype(int).astype(str) + ")")
    pilih = st.selectbox(
        "Pilih anime (1.500 terpopuler dari hasil filter)",
        options=kandidat.index.tolist(),
        format_func=lambda i: label_judul.loc[i])
    baris = kandidat.loc[pilih]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Skor", f"{baris['skor']:.0f} / 100")
    m2.metric("Popularitas", f"{int(baris['popularitas']):,}")
    m3.metric("Completion rate", f"{baris['completion_rate']*100:.1f}%")
    m4.metric("Drop rate", f"{baris['drop_rate']*100:.1f}%")

    s1, s2 = st.columns(2)
    df_f1 = pd.DataFrame({
        "Status": list(STATUS_KOLOM.keys()),
        "Pengguna": [int(baris[c]) for c in STATUS_KOLOM.values()],
    })
    fig_f1 = px.bar(df_f1, x="Pengguna", y="Status", orientation="h",
                    color="Status", color_discrete_map=WARNA_STATUS, text_auto=",",
                    title="Status Distribution")
    fig_f1.update_layout(showlegend=False)
    s1.plotly_chart(fig_f1, use_container_width=True)

    df_s1 = pd.DataFrame({"Bucket skor": BUCKET,
                          "Suara": [int(baris[c]) for c in SUARA_KOLOM]})
    fig_s1 = px.bar(df_s1, x="Bucket skor", y="Suara",
                    color="Bucket skor", color_continuous_scale="RdYlGn",
                    range_color=(10, 100), text_auto=",",
                    title="Score Distribution")
    fig_s1.update_layout(coloraxis_showscale=False)
    fig_s1.update_xaxes(tickmode="array", tickvals=BUCKET)
    s2.plotly_chart(fig_s1, use_container_width=True)

    st.divider()
    st.subheader("Genre dengan risiko ditinggalkan tertinggi")
    g = d[["genre", "drop_rate"]].dropna().copy()
    g["genre"] = g["genre"].str.split("|")
    g = g.explode("genre")
    agg = (g.groupby("genre")
             .agg(median_drop=("drop_rate", "median"), jumlah=("drop_rate", "size"))
             .query("jumlah >= 30")
             .sort_values("median_drop", ascending=False).head(10).reset_index())
    fig_g = px.bar(
        agg, x="median_drop", y="genre", orientation="h",
        labels={"median_drop": "Median drop rate", "genre": "Genre"},
        color="median_drop", color_continuous_scale="OrRd", text_auto=".1%",
        title="10 genre ber-drop-rate median tertinggi (min. 30 judul)")
    fig_g.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_g, use_container_width=True)

# ====================== TAB 4: MODEL & PREDIKSI ======================
with tab4:
    (model, fitur_final, fitur_dibuang, r2_uji, rmse_uji, mae_uji,
     vif_maks, TOP_GENRE, LEVEL) = latih_model(SIDIK)

    st.subheader("Model regresi linier berganda")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R² (data uji)", f"{r2_uji:.4f}")
    m2.metric("RMSE (data uji)", f"{rmse_uji:.2f} poin")
    m3.metric("MAE (data uji)", f"{mae_uji:.2f} poin")
    m4.metric("Prediktor", f"{len(fitur_final)}")
    m5.metric("Jumlah data", f"{len(df):,}")

    with st.expander("📋 Tabel koefisien model final (robust SE HC3)"):
        st.dataframe(tabel_koefisien(model), use_container_width=True, height=420)

    st.divider()
    st.subheader("🔮 Coba prediksi skor anime")

    i1, i2, i3 = st.columns(3)
    in_pop = i1.number_input("Popularitas (pengguna yang menyimpan)",
                             min_value=10, max_value=5_000_000,
                             value=int(df["popularitas"].median()), step=500)
    in_fav = i2.number_input("Jumlah favorit", min_value=0, max_value=1_000_000,
                             value=int(df["favorit"].median()), step=100)
    in_eps = i3.number_input("Jumlah episode", min_value=1, max_value=500,
                             value=int(df["episodes"].median()))

    i4, i5, i6 = st.columns(3)
    in_dur = i4.number_input("Durasi per episode (menit)", min_value=1, max_value=200,
                             value=int(df["durasi_menit"].median()))
    in_thn = i5.number_input("Tahun rilis", min_value=th_min, max_value=th_max,
                             value=min(2024, th_max))
    in_format = i6.selectbox("Format", LEVEL["format"],
                             index=LEVEL["format"].index("TV"))

    i7, i8, i9 = st.columns(3)
    in_sumber = i7.selectbox("Sumber adaptasi", LEVEL["sumber_adaptasi"],
                             index=LEVEL["sumber_adaptasi"].index("ORIGINAL"))
    in_negara = i8.selectbox("Negara asal", LEVEL["negara_asal"],
                             index=LEVEL["negara_asal"].index("JP"))
    in_genre = i9.multiselect("Genre (identitas + jumlah)", semua_genre,
                              default=[g for g in ("Comedy",) if g in semua_genre],
                              placeholder="Pilih genre")

    i10, i11 = st.columns(2)
    in_drop = i10.slider("Drop rate (%) — penonton yang berhenti", 0, 100,
                         int(df["drop_rate"].median() * 100)) / 100
    in_plan = i11.slider("Planning ratio (%) — niat yang belum terkonversi", 0, 100,
                         int(df["planning_ratio"].median() * 100)) / 100

    if st.button("Hitung prediksi skor", type="primary"):
        nilai_numerik = {
            "log_popularitas": np.log10(in_pop),
            "rasio_favorit": min(in_fav / max(in_pop, 1), 1.0),
            "log_episodes": np.log10(in_eps + 1),
            "durasi_menit": in_dur,
            "tahun_sejak_2000": in_thn - 2000,
            "jumlah_genre": len(in_genre),
            "drop_rate": in_drop,
            "planning_ratio": in_plan,
        }
        baris_in = rakit_input_prediksi(fitur_final, nilai_numerik,
                                        in_format, in_sumber, in_negara, in_genre)
        prediksi = float(model.predict(baris_in).iloc[0])
        prediksi = min(max(prediksi, 0), 100)

        st.success(f"### Prediksi skor: **{prediksi:.1f} / 100**")
        st.caption(
            f"Perkiraan rentang wajar: {max(prediksi - rmse_uji, 0):.1f} – "
            f"{min(prediksi + rmse_uji, 100):.1f} (± RMSE data uji).")

st.divider()
st.caption(
    "Final Project Big Data & Predictive Analytics."
)

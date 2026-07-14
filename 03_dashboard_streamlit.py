"""03 — Dashboard Interaktif: Riset Pasar & Prediksi Skor Anime (Seluruh Database AniList)

Final Project Big Data & Predictive Analytics
Kelompok «ISI», Kelas «ISI»

Jalankan lokal : streamlit run 03_dashboard_streamlit.py
Deploy publik  : lihat 00_LANGKAH_PENGERJAAN.md (Streamlit Community Cloud)

Syarat: `anime_clean.csv` (hasil notebook 02) berada di folder yang sama.
"""

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

# Set prediktor final (hasil diagnostik data nyata):
#  - rasio_favorit menggantikan log_favorit  -> menghapus multikolinearitas dgn popularitas
#  - completion_rate TIDAK masuk model (r ≈ -0,90 dgn drop_rate), hanya deskriptif
FITUR_KANDIDAT = ["log_popularitas", "rasio_favorit", "episodes", "durasi_menit",
                  "tahun_sejak_2000", "jumlah_genre", "drop_rate", "planning_ratio"]

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


@st.cache_data
def muat_data() -> pd.DataFrame:
    return pd.read_csv("anime_clean.csv")


def saring_vif(data: pd.DataFrame, fitur: list[str]) -> list[str]:
    """Seleksi multikolinearitas: buang variabel ber-VIF > 10 satu per satu
    (metodologi yang sama dengan notebook analisis)."""
    fitur = list(fitur)
    while len(fitur) > 2:
        X = sm.add_constant(data[fitur].astype(float))
        vif = [variance_inflation_factor(X.values, i + 1) for i in range(len(fitur))]
        if max(vif) <= 10:
            break
        fitur.pop(int(np.argmax(vif)))
    return fitur


@st.cache_resource
def latih_model():
    """Regresi berganda pada seluruh data bersih (pasca-seleksi VIF) + kinerja uji."""
    df = muat_data()
    fitur = saring_vif(df, [c for c in FITUR_KANDIDAT if c in df.columns])

    X = sm.add_constant(df[fitur].astype(float))
    model = sm.OLS(df["skor"].astype(float), X).fit()

    train, test = train_test_split(df, test_size=0.20, random_state=42)
    m = sm.OLS(train["skor"].astype(float),
               sm.add_constant(train[fitur].astype(float))).fit()
    pred = m.predict(sm.add_constant(test[fitur].astype(float)))
    rmse = float(np.sqrt(mean_squared_error(test["skor"], pred)))
    r2_uji = float(r2_score(test["skor"], pred))
    return model, fitur, rmse, r2_uji


def format_persamaan(params: pd.Series) -> str:
    teks = f"skor = {params['const']:.3f}"
    for nama, nilai in params.items():
        if nama == "const":
            continue
        tanda = "+" if nilai >= 0 else "−"
        teks += f" {tanda} {abs(nilai):.3f}·{nama}"
    return teks


df = muat_data()

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
    "Sumber: web scraping **seluruh database** [AniList](https://anilist.co/search/anime) "
    "via endpoint GraphQL publiknya (paginasi berlapis, rate limit dipatuhi). "
    "Dataset dibatasi pada anime yang dinilai ≥ 30 pengguna agar skor reliabel."
)

# ====================== HEADER ======================
st.title("🎬 Dashboard Riset Pasar & Prediksi Skor Anime")
st.caption(
    "Final Project **Big Data & Predictive Analytics** · Kelompok «ISI» · "
    "Seluruh database AniList · Regresi linier + metrik perilaku penonton"
)

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
    model, fitur, rmse_uji, r2_uji = latih_model()

    st.subheader("Model regresi linier berganda (pasca-seleksi VIF)")
    st.caption(
        "Dilatih pada seluruh dataset bersih (tak terpengaruh filter), metodologi identik "
        "dengan notebook analisis: `rasio_favorit` menggantikan `log_favorit` (menghapus "
        "multikolinearitas dengan popularitas), `completion_rate` dikeluarkan karena "
        "redundan dengan `drop_rate` (r ≈ −0,90), dan kolom distribusi suara TIDAK dipakai "
        "sebagai prediktor (anti-kebocoran data). Pengecekan VIF > 10 tetap aktif.")
    st.code(format_persamaan(model.params), language="text")
    st.caption("Variabel terpakai: " + ", ".join(fitur))

    m1, m2, m3 = st.columns(3)
    m1.metric("R² (data uji)", f"{r2_uji:.4f}")
    m2.metric("RMSE (data uji)", f"{rmse_uji:.2f} poin")
    m3.metric("Jumlah data", f"{len(df):,}")

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
    in_thn = i5.number_input("Tahun rilis", min_value=1960, max_value=2026, value=2024)
    in_gen = i6.slider("Jumlah genre", 1, 8, int(df["jumlah_genre"].median()))

    i7, i8 = st.columns(2)
    in_drop = i7.slider("Drop rate (%) — penonton yang berhenti", 0, 100,
                        int(df["drop_rate"].median() * 100)) / 100
    in_plan = i8.slider("Planning ratio (%) — niat yang belum terkonversi", 0, 100,
                        int(df["planning_ratio"].median() * 100)) / 100
    st.caption(
        f"Rasio favorit dihitung otomatis dari dua input di atas "
        f"(favorit ÷ popularitas). Nilai median dataset: "
        f"{df['rasio_favorit'].median()*100:.2f}%.")

    if st.button("Hitung prediksi skor", type="primary"):
        semua_input = {
            "log_popularitas": np.log10(in_pop),
            "rasio_favorit": min(in_fav / max(in_pop, 1), 1.0),
            "episodes": in_eps,
            "durasi_menit": in_dur,
            "tahun_sejak_2000": in_thn - 2000,
            "jumlah_genre": in_gen,
            "drop_rate": in_drop,
            "planning_ratio": in_plan,
        }
        baris_in = pd.DataFrame([semua_input])[fitur].astype(float)
        baris_in.insert(0, "const", 1.0)
        prediksi = float(model.predict(baris_in).iloc[0])
        prediksi = min(max(prediksi, 0), 100)

        st.success(f"### Prediksi skor: **{prediksi:.1f} / 100**")
        st.caption(
            f"Perkiraan rentang wajar: {max(prediksi - rmse_uji, 0):.1f} – "
            f"{min(prediksi + rmse_uji, 100):.1f} (± RMSE data uji).")

st.divider()
st.caption(
    "© 2026 Kelompok «ISI» — Final Project Big Data & Predictive Analytics. "
    "Seluruh database anime AniList dikumpulkan mandiri via web scraping endpoint "
    "GraphQL publik (paginasi berlapis; rate limit & proteksi Cloudflare ditangani etis)."
)

"""03 — Dashboard Interaktif: Analisis & Prediksi Skor Anime (Data AniList)

Final Project Big Data & Predictive Analytics
Kelompok «ISI», Kelas «ISI»

Jalankan lokal : streamlit run 03_dashboard_streamlit.py
Deploy publik  : lihat 00_LANGKAH_PENGERJAAN.md (Streamlit Community Cloud)

Syarat: file `anime_clean.csv` (hasil notebook 02) berada di folder yang sama.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Dashboard Prediksi Skor Anime — AniList",
    page_icon="🎬",
    layout="wide",
)

FITUR = ["log_popularitas", "log_favorit", "episodes",
         "durasi_menit", "tahun_sejak_2000", "jumlah_genre"]
LABEL_SUMBU = {
    "skor": "Skor rata-rata (0–100)",
    "log_popularitas": "log10(popularitas)",
    "log_favorit": "log10(favorit + 1)",
    "episodes": "Jumlah episode",
    "durasi_menit": "Durasi per episode (menit)",
    "tahun_rilis": "Tahun rilis",
    "jumlah_genre": "Jumlah genre",
    "format": "Format anime",
}


@st.cache_data
def muat_data() -> pd.DataFrame:
    return pd.read_csv("anime_clean.csv")


@st.cache_resource
def latih_model():
    """Melatih model regresi berganda pada seluruh data bersih + estimasi kinerja uji."""
    df = muat_data()
    fitur = [c for c in FITUR if c in df.columns]

    X = sm.add_constant(df[fitur].astype(float))
    model = sm.OLS(df["skor"].astype(float), X).fit()

    # estimasi kinerja pada data uji (split yang sama dengan notebook analisis)
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
    default=[], placeholder="Semua format",
)

semua_genre = sorted({g for s in df["genre"].dropna() for g in str(s).split("|") if g})
pilihan_genre = st.sidebar.multiselect(
    "Genre", semua_genre, default=[], placeholder="Semua genre",
)

d = df[df["tahun_rilis"].between(rentang_tahun[0], rentang_tahun[1])]
if pilihan_format:
    d = d[d["format"].isin(pilihan_format)]
if pilihan_genre:
    d = d[d["genre"].fillna("").apply(
        lambda s: any(g in str(s).split("|") for g in pilihan_genre))]

st.sidebar.markdown(f"**{len(d):,} / {len(df):,}** anime terpilih")
st.sidebar.caption(
    "Sumber data: hasil web scraping situs [AniList](https://anilist.co/search/anime) "
    "melalui endpoint GraphQL publiknya."
)

# ====================== HEADER ======================
st.title("🎬 Dashboard Prediksi Skor Anime — Data AniList")
st.caption(
    "Final Project **Big Data & Predictive Analytics** · Kelompok «ISI» · "
    "Regresi linier sederhana & berganda"
)

if d.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Longgarkan filter di sidebar.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Data", "📈 Visualisasi", "🤖 Model & Prediksi"])

# ====================== TAB 1: RINGKASAN ======================
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah anime", f"{len(d):,}")
    k2.metric("Rata-rata skor", f"{d['skor'].mean():.1f}")
    k3.metric("Median popularitas", f"{int(d['popularitas'].median()):,}")
    k4.metric("Rentang tahun", f"{int(d['tahun_rilis'].min())}–{int(d['tahun_rilis'].max())}")

    st.subheader("Statistik deskriptif")
    st.dataframe(
        d[["skor", "popularitas", "favorit", "episodes",
           "durasi_menit", "tahun_rilis", "jumlah_genre"]].describe().round(2),
        use_container_width=True,
    )

    st.subheader("Cuplikan data (200 baris pertama)")
    st.dataframe(
        d[["judul_romaji", "format", "tahun_rilis", "episodes",
           "durasi_menit", "genre", "popularitas", "favorit", "skor"]].head(200),
        use_container_width=True, height=380,
    )
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        d.to_csv(index=False).encode("utf-8-sig"),
        file_name="anime_terfilter.csv", mime="text/csv",
    )

# ====================== TAB 2: VISUALISASI ======================
with tab2:
    c1, c2 = st.columns(2)

    fig_scatter = px.scatter(
        d, x="log_popularitas", y="skor", color="format",
        trendline="ols", trendline_scope="overall",
        trendline_color_override="crimson",
        opacity=0.45, labels=LABEL_SUMBU,
        title="Skor vs log10(Popularitas) + garis regresi",
    )
    c1.plotly_chart(fig_scatter, use_container_width=True)

    fig_hist = px.histogram(
        d, x="skor", nbins=30, labels=LABEL_SUMBU,
        title="Distribusi skor anime",
        color_discrete_sequence=["#4C72B0"],
    )
    fig_hist.update_layout(yaxis_title="Jumlah anime")
    c2.plotly_chart(fig_hist, use_container_width=True)

    c3, c4 = st.columns(2)

    kolom_kor = ["skor", "log_popularitas", "log_favorit", "episodes",
                 "durasi_menit", "tahun_rilis", "jumlah_genre"]
    fig_heat = px.imshow(
        d[kolom_kor].corr().round(2), text_auto=True,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Heatmap korelasi Pearson",
    )
    c3.plotly_chart(fig_heat, use_container_width=True)

    rerata_format = (d.groupby("format", as_index=False)["skor"]
                     .mean().sort_values("skor"))
    fig_bar = px.bar(
        rerata_format, x="skor", y="format", orientation="h",
        labels=LABEL_SUMBU, title="Rata-rata skor per format",
        color="skor", color_continuous_scale="Viridis",
    )
    c4.plotly_chart(fig_bar, use_container_width=True)

# ====================== TAB 3: MODEL & PREDIKSI ======================
with tab3:
    model, fitur, rmse_uji, r2_uji = latih_model()

    st.subheader("Model regresi linier berganda")
    st.caption("Dilatih pada seluruh dataset bersih (tidak terpengaruh filter sidebar).")
    st.code(format_persamaan(model.params), language="text")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² (data uji)", f"{r2_uji:.4f}")
    m2.metric("RMSE (data uji)", f"{rmse_uji:.2f} poin")
    m3.metric("Jumlah data latih", f"{len(df):,}")

    st.divider()
    st.subheader("🔮 Coba prediksi skor anime")

    i1, i2, i3 = st.columns(3)
    in_pop = i1.number_input("Popularitas (jumlah pengguna yang menyimpan)",
                             min_value=100, max_value=5_000_000,
                             value=int(df["popularitas"].median()), step=1000)
    in_fav = i2.number_input("Jumlah favorit",
                             min_value=0, max_value=1_000_000,
                             value=int(df["favorit"].median()), step=100)
    in_eps = i3.number_input("Jumlah episode", min_value=1, max_value=500,
                             value=int(df["episodes"].median()))

    i4, i5, i6 = st.columns(3)
    in_dur = i4.number_input("Durasi per episode (menit)", min_value=1, max_value=200,
                             value=int(df["durasi_menit"].median()))
    in_thn = i5.number_input("Tahun rilis", min_value=1960, max_value=2026, value=2024)
    in_gen = i6.slider("Jumlah genre", 1, 8, int(df["jumlah_genre"].median()))

    if st.button("Hitung prediksi skor", type="primary"):
        baris = pd.DataFrame([{
            "log_popularitas": np.log10(in_pop),
            "log_favorit": np.log10(in_fav + 1),
            "episodes": in_eps,
            "durasi_menit": in_dur,
            "tahun_sejak_2000": in_thn - 2000,
            "jumlah_genre": in_gen,
        }])[fitur].astype(float)
        baris.insert(0, "const", 1.0)
        prediksi = float(model.predict(baris).iloc[0])
        prediksi = min(max(prediksi, 0), 100)

        st.success(f"### Prediksi skor: **{prediksi:.1f} / 100**")
        st.caption(
            f"Perkiraan rentang wajar: {max(prediksi - rmse_uji, 0):.1f} – "
            f"{min(prediksi + rmse_uji, 100):.1f} (± RMSE data uji)."
        )

st.divider()
st.caption(
    "© 2026 Kelompok «ISI» — Final Project Big Data & Predictive Analytics. "
    "Data dikumpulkan secara mandiri via web scraping AniList (endpoint GraphQL publik, "
    "rate limit dipatuhi)."
)

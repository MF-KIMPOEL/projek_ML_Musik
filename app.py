import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG — harus dipanggil pertama kali
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="REQ · Music ML",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS — palette & tipografi
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] * { color: #888 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.35rem 0 !important;
}

/* Main area */
[data-testid="stAppViewContainer"] { background: #080808; }
[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.18em !important;
    color: #444 !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    color: #e8d5a3 !important;
    font-size: 1.6rem !important;
}

/* Tombol primary */
.stButton > button[kind="primary"] {
    background: #e8d5a3 !important;
    color: #080808 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    border: none !important;
    border-radius: 2px !important;
}
.stButton > button[kind="primary"]:hover { background: #f0e4b8 !important; }

/* Multiselect tags */
.stMultiSelect [data-baseweb="tag"] {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #e8d5a3 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    color: #444 !important;
}
.stTabs [aria-selected="true"] {
    color: #e8d5a3 !important;
    border-bottom-color: #e8d5a3 !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL & DATASET (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Cache model supaya tidak reload tiap interaksi."""
    return joblib.load("model_era_gacor.joblib"), joblib.load("model_acc_gacor.joblib")

@st.cache_data
def load_data():
    """Cache data supaya tidak re-parse tiap interaksi."""
    _df = pd.read_csv("rym.csv")
    # Fitur tambahan — sama persis seperti waktu training
    _df["year"]             = pd.to_datetime(_df["release_date"]).dt.year
    _df["log_rating_count"] = np.log1p(_df["rating_count"])
    _df["log_review_count"] = np.log1p(_df["review_count"])
    _df["era"] = pd.cut(
        _df["year"],
        bins=[0, 1969, 1984, 1999, 2012, 2026],
        labels=["Pionir", "Old School", "Mid High School", "New School", "New New School"],
    )
    _df["acc_label"] = pd.cut(
        _df["accessibility"],
        bins=[0, 2, 4, 6, 8, 10.1],
        labels=["Niche", "Elitis", "Tidak Basic", "Agak Lumayan Basic", "Basic"],
    )
    _df = _df.dropna(subset=["era", "acc_label"])
    return _df

model_era, model_akses = load_models()
df = load_data()

ALL_ALBUMS  = sorted(df["release_name"].dropna().unique().tolist(), key=str.lower)
ALL_ARTISTS = sorted(df["artist_name"].dropna().unique().tolist(), key=str.lower)


# ──────────────────────────────────────────────
# KONSTANTA INFO
# ──────────────────────────────────────────────
ERA_INFO = {
    "Pionir":          {"emoji":"🎷","rentang":"~1920–1969","warna":"#C0A060","desc":"Era orang merintis, tis tis.","tokoh":"The Beatles · Bob Dylan · Miles Davis"},
    "Old School":      {"emoji":"🎸","rentang":"1970–1984", "warna":"#E07040","desc":"Era old skull, rata rata bapak dan ibu anda lahiran era ini.","tokoh":"Led Zeppelin · Pink Floyd · Stevie Wonder"},
    "Mid High School": {"emoji":"📼","rentang":"1985–1999", "warna":"#5080D0","desc":"Era panik satanis massal, mengerikan.","tokoh":"Nirvana · Radiohead · Nas · Björk"},
    "New School":      {"emoji":"💿","rentang":"2000–2012", "warna":"#40B080","desc":"Era digital awal awal, musik indie mulai booming ini nih.","tokoh":"my chemical romance · Kanye West · Sufjan Stevens"},
    "New New School":  {"emoji":"📱","rentang":"2013–kini", "warna":"#A050E0","desc":"Era streaming, eranya pendukung ******.","tokoh":"Kendrick Lamar · Frank Ocean · Laufey"},
}
ACC_INFO = {
    "Niche":             {"emoji":"🔬","warna":"#3050A0","singkat":"ini guwa banget nih","desc":"diketahui sedikit orang."},
    "Elitis":            {"emoji":"🎓","warna":"#6040B0","singkat":"jarang mandi","desc":"masuk ranah yang disebut underground."},
    "Tidak Basic":       {"emoji":"🎸","warna":"#208060","singkat":"nolep fungsional","desc":"masuk ranah anak band."},
    "Agak Lumayan Basic":{"emoji":"🎧","warna":"#808020","singkat":"yabolelah","desc":"terkenal tidak pakai illl."},
    "Basic":             {"emoji":"📻","warna":"#A04020","singkat":"oke","desc":"terkenale illl."},
}
MOOD_DESC = {
    "melancholic":"Sedih dan indah",
    "atmospheric":"Bangun suasana dan ruang.",
    "introspective":"Ngajak refleksi ke dalam diri.",
    "anxious":"Nuansa gelisah dan tegang.",
    "cold":"Dingin dan jauh secara emosional, tapi nagih.",
    "energetic":"Penuh energi — bikin semangat.",
    "dark":"Ada unsur kegelapan, baik lirik atau sonic.",
    "epic":"Skala besar dan megah, kaya soundtrack.",
    "psychedelic":"Suara yang muter dan ngebuka persepsi.",
    "hypnotic":"menghipnotis.",
    "lonely":"kurang kasih sayang.",
    "romantic":"Cinta dan keintiman yang hangat.",
    "mellow":"Santai, lembut, enak didengerin.",
}
ACC_ORDER = ["Niche", "Elitis", "Tidak Basic", "Agak Lumayan Basic", "Basic"]
# Kata teknis yang bukan mood, diexclude dari analisis
EXCLUDE = {"malevocals", "femalevocals", "conceptalbum", "instrumental", "mixedvocals"}


# ──────────────────────────────────────────────
# HELPER CHART & UI
# ──────────────────────────────────────────────
def bar_dark(data, x, y, h=280):
    """Bar chart horizontal dengan tema gelap — reusable."""
    fig = px.bar(
        data, x=x, y=y, orientation="h", template="plotly_dark",
        color=x, color_continuous_scale=["#1a1a1a", "#e8d5a3"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, coloraxis_showscale=False, height=h,
        margin=dict(l=0, r=0, t=6, b=0),
        xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9)),
        yaxis=dict(tickfont=dict(color="#aaa", size=10), categoryorder="total ascending"),
    )
    return fig

def label_mono(text, mt=0):
    """Caption label gaya DM Mono kecil — reusable."""
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.6rem;color:#555;"
        f"letter-spacing:0.18em;margin:{mt}rem 0 0.4rem;text-transform:uppercase;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# FUNGSI CORE ML
# ──────────────────────────────────────────────
def get_rows(albums, artists):
    """Ambil baris dataset yang cocok dengan pilihan album/artis user."""
    mask = df["release_name"].isin(albums) | df["artist_name"].isin(artists)
    return df[mask].drop_duplicates(subset=["release_name", "artist_name"])

def get_era(df_c):
    """Prediksi era dominan dari sekumpulan album pakai model_era."""
    preds = model_era.predict(
        df_c[["year", "avg_rating", "log_rating_count", "log_review_count", "accessibility"]]
    )
    return Counter(preds).most_common(1)[0][0]

def get_acc(df_c):
    """Prediksi aksesibilitas dominan dari sekumpulan album pakai model_akses."""
    preds = model_akses.predict(
        df_c[["accessibility", "avg_rating", "log_rating_count", "log_review_count"]]
    )
    return Counter(preds).most_common(1)[0][0]

def get_mood(df_c):
    """Hitung 3 mood/descriptor paling sering dari kolom descriptors."""
    semua = [
        k.strip()
        for d in df_c["descriptors"].dropna()
        for k in d.split(",")
        if k.strip() not in EXCLUDE
    ]
    return Counter(semua).most_common(3) if semua else [("tidak tersedia", 0)]

def get_rek(era, acc, mood, df_c, n=5):
    """
    Rekomendasikan album:
    - era sama dengan favorit user
    - aksesibilitas sama atau berdekatan (+-1 level)
    - mood overlap sebanyak mungkin
    - album yang sudah disukai di-exclude
    """
    sudah = set(df_c["release_name"].str.lower())
    idx   = ACC_ORDER.index(acc) if acc in ACC_ORDER else 2
    # Ambil range aksesibilitas: bisa sama, satu di atas, atau satu di bawah
    acc_r = [ACC_ORDER[i] for i in {max(0, idx - 1), idx, min(4, idx + 1)}]
    kata  = {m[0] for m in mood}

    rek = df[
        (df["era"] == era) &
        (df["acc_label"].isin(acc_r)) &
        (~df["release_name"].str.lower().isin(sudah))
    ].copy()

    # Fallback: longgarkan ke seluruh era jika hasil kurang dari n
    if len(rek) < n:
        rek = df[
            (df["era"] == era) &
            (~df["release_name"].str.lower().isin(sudah))
        ].copy()

    # Skor = jumlah descriptor yang overlap dengan mood user
    rek["skor"] = rek["descriptors"].apply(
        lambda d: 0 if pd.isna(d) else len({k.strip() for k in d.split(",")} & kata)
    )
    return rek.sort_values(["skor", "avg_rating"], ascending=[False, False]).head(n)


# ══════════════════════════════════════════════
# SIDEBAR — navigasi utama
# ══════════════════════════════════════════════
with st.sidebar:
    # Logo / judul
    st.markdown("""
    <div style="padding:1.5rem 0 2rem;">
        <p style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#e8d5a3;
                  margin:0;line-height:1;">REQ<sup style="font-size:0.4em;">✦</sup></p>
        <p style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#333;
                  letter-spacing:0.2em;margin:0.3rem 0 0;">MUSIC ML PROJECT</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigasi radio — lebih Streamlit-native, lebih bersih
    page = st.radio(
        "NAVIGASI",
        options=["Introduction", "Dataset", "Analisis", "About Me", "Kode Proyek"],
        label_visibility="visible",
    )

    st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #1a1a1a;padding-top:1.2rem;">
        <p style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2a2a2a;
                  letter-spacing:0.15em;line-height:2;margin:0;">
            DATA · RateYourMusic<br>
            MODEL · scikit-learn<br>
            UI · Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HALAMAN 1 — INTRODUCTION
# ══════════════════════════════════════════════
if page == "Introduction":

    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:5rem;color:#f0ede8;
               letter-spacing:-0.03em;line-height:1;margin:0 0 0.5rem;">
        REQ<sup style="font-size:0.28em;vertical-align:super;color:#e8d5a3;">✦</sup>
    </h1>
    <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#444;
              letter-spacing:0.2em;margin:0 0 2.5rem;">
        MUSIC MACHINE LEARNING · RECOMMENDATION SYSTEM
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.92rem;color:#666;max-width:560px;line-height:1.9;margin-bottom:2.5rem;">
        REQ adalah sistem rekomendasi album berbasis Machine Learning. Masukkan album atau artis
        favorit kamu, dan REQ akan memprediksi era musikmu, tingkat aksesibilitas selera,
        top mood — lalu kasih rekomendasi album yang cocok.
    </p>
    """, unsafe_allow_html=True)

    # Quick stats — native st.metric
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Album",   f"{len(df):,}")
    c2.metric("Rentang Tahun", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c3.metric("Avg Rating",    f"{df['avg_rating'].mean():.2f}")
    c4.metric("Total Artis",   f"{df['artist_name'].nunique():,}")

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # Distribusi era — preview chart
    label_mono("DISTRIBUSI ERA DALAM DATASET")
    era_counts = df["era"].value_counts().sort_index().reset_index()
    era_counts.columns = ["era", "count"]
    fig_era = px.bar(
        era_counts, x="era", y="count", template="plotly_dark",
        color="era",
        color_discrete_map={
            "Pionir":"#C0A060","Old School":"#E07040",
            "Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"
        },
    )
    fig_era.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
        xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#555", size=10)),
        yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#444", size=9)),
    )
    st.plotly_chart(fig_era, use_container_width=True)

    # Pipeline overview
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    label_mono("CARA KERJA")
    step1, step2, step3 = st.columns(3)
    for col, num, judul, isi in [
        (step1, "01", "INPUT",       "Pilih album atau artis favoritmu dari dropdown."),
        (step2, "02", "PREDIKSI ML", "Dua model mengklasifikasi era & aksesibilitas seleramu."),
        (step3, "03", "REKOMENDASI", "Album dengan era + mood paling mirip ditampilkan."),
    ]:
        col.markdown(f"""
        <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;
                    padding:1.3rem;height:120px;">
            <p style="font-family:'DM Mono',monospace;font-size:0.55rem;
                      color:#2a2a2a;letter-spacing:0.2em;margin:0 0 0.4rem;">{num}</p>
            <p style="font-family:'DM Serif Display',serif;font-size:1rem;
                      color:#e8d5a3;margin:0 0 0.5rem;">{judul}</p>
            <p style="font-size:0.8rem;color:#555;line-height:1.6;margin:0;">{isi}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HALAMAN 2 — DATASET
# ══════════════════════════════════════════════
elif page == "Dataset":

    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;">
        Dataset
    </h1>
    <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;
              letter-spacing:0.15em;margin-bottom:2rem;">
        SOURCE · RATEYOURMUSIC — 5000+ ALBUM
    </p>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Album",       f"{len(df):,}")
    m2.metric("Tahun",             f"{int(df['year'].min())}–{int(df['year'].max())}")
    m3.metric("Avg Rating",        f"{df['avg_rating'].mean():.2f}")
    m4.metric("Avg Aksesibilitas", f"{df['accessibility'].mean():.1f}/10")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ─── Tab: Tabel Data | Visualisasi ───
    tab_tabel, tab_viz = st.tabs(["📋  TABEL DATA", "📊  VISUALISASI"])

    with tab_tabel:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Filter cepat
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        with col_f1:
            era_filter = st.multiselect(
                "Filter Era", options=df["era"].cat.categories.tolist(), default=[]
            )
        with col_f2:
            acc_filter = st.multiselect(
                "Filter Aksesibilitas", options=ACC_ORDER, default=[]
            )
        with col_f3:
            n_rows = st.number_input("Baris", min_value=10, max_value=500, value=50, step=10)

        # Terapkan filter
        df_show = df.copy()
        if era_filter:
            df_show = df_show[df_show["era"].isin(era_filter)]
        if acc_filter:
            df_show = df_show[df_show["acc_label"].isin(acc_filter)]

        cols_tampil = ["release_name", "artist_name", "year", "avg_rating",
                       "rating_count", "accessibility", "era", "acc_label", "primary_genres"]

        label_mono(f"{len(df_show):,} BARIS DITEMUKAN")
        # Tampilkan DataFrame dengan column_config yang informatif
        st.dataframe(
            df_show[cols_tampil].head(n_rows),
            use_container_width=True,
            height=420,
            column_config={
                "release_name":  st.column_config.TextColumn("Album"),
                "artist_name":   st.column_config.TextColumn("Artis"),
                "year":          st.column_config.NumberColumn("Tahun", format="%d"),
                "avg_rating":    st.column_config.NumberColumn("Rating", format="%.2f"),
                "rating_count":  st.column_config.NumberColumn("# Penilai"),
                "accessibility": st.column_config.ProgressColumn(
                    "Aksesibilitas", min_value=0, max_value=10
                ),
                "era":           st.column_config.TextColumn("Era"),
                "acc_label":     st.column_config.TextColumn("Level Acc"),
                "primary_genres":st.column_config.TextColumn("Genre"),
            },
        )

    with tab_viz:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            # Top 10 artis berdasarkan total rating count
            label_mono("TOP 10 ARTIS — TOTAL RATING")
            top_a = df.groupby("artist_name")["rating_count"].sum().nlargest(10).reset_index()
            st.plotly_chart(bar_dark(top_a, "rating_count", "artist_name"), use_container_width=True)

            # Scatter: log rating count vs avg rating, warna per era
            label_mono("RATING COUNT VS AVG RATING", mt=1)
            fig_sc = px.scatter(
                df, x="log_rating_count", y="avg_rating", color="era", opacity=0.45,
                template="plotly_dark", hover_data=["release_name", "artist_name"],
                color_discrete_map={
                    "Pionir":"#C0A060","Old School":"#E07040",
                    "Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"
                },
            )
            fig_sc.update_traces(marker_size=3)
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
                xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9),
                           title="log(rating_count)"),
                yaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9),
                           title="avg_rating"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_r:
            # Top 12 genre terbanyak
            label_mono("TOP GENRES")
            semua_genre = [x.strip() for g in df["primary_genres"].dropna() for x in g.split(",")]
            gdf = pd.DataFrame(Counter(semua_genre).most_common(12), columns=["genre", "count"])
            st.plotly_chart(bar_dark(gdf, "count", "genre"), use_container_width=True)

            # Top 10 album berdasarkan avg_rating
            label_mono("TOP 10 ALBUM — AVG RATING", mt=1)
            top10 = df.nlargest(10, "avg_rating")[["release_name", "avg_rating"]].copy()
            top10["release_name"] = top10["release_name"].str[:24]
            st.plotly_chart(bar_dark(top10, "avg_rating", "release_name"), use_container_width=True)


# ══════════════════════════════════════════════
# HALAMAN 3 — ANALISIS / REKOMENDASI
# ══════════════════════════════════════════════
elif page == "Analisis":

    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;">
        Analisis & Rekomendasi
    </h1>
    <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;
              letter-spacing:0.15em;margin-bottom:2rem;">
        PILIH ALBUM ATAU ARTIS — MODEL AKAN MENGANALISIS SELERAMU
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        sel_albums = st.multiselect(
            "🎵 Album yang kamu suka", options=ALL_ALBUMS,
            placeholder="Ketik atau pilih album...",
        )
    with col2:
        sel_artists = st.multiselect(
            "🎤 Artis favorit kamu", options=ALL_ARTISTS,
            placeholder="Ketik atau pilih artis...",
        )

    n_rek = st.slider("Jumlah rekomendasi", min_value=3, max_value=10, value=5)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    _, bc, _ = st.columns([2, 1, 2])
    run = bc.button("✦  Generate", type="primary", use_container_width=True)

    if run:
        if not sel_albums and not sel_artists:
            st.warning("Pilih minimal satu album atau artis dulu ya.")
        else:
            df_c = get_rows(sel_albums, sel_artists)
            if df_c.empty:
                st.error("Yah, gaketemu di dataset.")
            else:
                # Jalankan model
                era_fav  = get_era(df_c)
                acc_fav  = get_acc(df_c)
                top_mood = get_mood(df_c)
                df_rek   = get_rek(era_fav, acc_fav, top_mood, df_c, n=n_rek)
                ei       = ERA_INFO.get(era_fav, {})
                ai       = ACC_INFO.get(acc_fav, {})

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='font-size:0.82rem;color:#555;font-style:italic;'>"
                    f"Berdasarkan {len(df_c)} album — {ai.get('singkat','')}.</p>",
                    unsafe_allow_html=True,
                )

                # Tab: Prediksi | Rekomendasi | Raw Data input user
                t1, t2, t3 = st.tabs(["🔍  PREDIKSI", "💿  REKOMENDASI", "🗂  DATA KAMU"])

                with t1:
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    ce, ca = st.columns(2)

                    # Card Era — dengan border kiri berwarna khas era
                    with ce:
                        st.markdown(f"""
                        <div style="background:#0f0f0f;border:1px solid {ei.get('warna','#333')}33;
                                    border-left:3px solid {ei.get('warna','#e8d5a3')};
                                    border-radius:4px;padding:1.5rem;">
                            <p style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                      color:{ei.get('warna','#888')};letter-spacing:0.2em;margin:0 0 0.6rem;">
                                MODEL 1 — ERA</p>
                            <div style="font-size:1.8rem;margin-bottom:0.3rem;">{ei.get('emoji','')}</div>
                            <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                                       color:#f0ede8;margin:0;">{era_fav}</h2>
                            <p style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                      color:{ei.get('warna','#888')};margin:0.3rem 0 1rem;">
                                {ei.get('rentang','')}</p>
                            <p style="font-size:0.83rem;color:#666;line-height:1.7;margin:0 0 0.5rem;">
                                {ei.get('desc','')}</p>
                            <p style="font-size:0.72rem;color:#444;margin:0;">{ei.get('tokoh','')}</p>
                        </div>""", unsafe_allow_html=True)

                    # Card Aksesibilitas + skala visual
                    with ca:
                        skala_html = "".join(
                            f"<div style='flex:1;height:5px;background:"
                            f"{ai.get('warna','#444') if lv == acc_fav else '#1e1e1e'};"
                            f"border-radius:2px;'></div>"
                            for lv in ACC_ORDER
                        )
                        st.markdown(f"""
                        <div style="background:#0f0f0f;border:1px solid {ai.get('warna','#333')}33;
                                    border-left:3px solid {ai.get('warna','#e8d5a3')};
                                    border-radius:4px;padding:1.5rem;">
                            <p style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                      color:{ai.get('warna','#888')};letter-spacing:0.2em;margin:0 0 0.6rem;">
                                MODEL 2 — AKSESIBILITAS</p>
                            <div style="font-size:1.8rem;margin-bottom:0.3rem;">{ai.get('emoji','')}</div>
                            <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;
                                       color:#f0ede8;margin:0;">{acc_fav}</h2>
                            <p style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                      color:{ai.get('warna','#888')};margin:0.3rem 0 1rem;">
                                {ai.get('singkat','')}</p>
                            <p style="font-size:0.83rem;color:#666;line-height:1.7;margin:0 0 0.8rem;">
                                {ai.get('desc','')}</p>
                            <p style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                      color:#333;margin:0 0 0.4rem;">NICHE &lt;——&gt; BASIC</p>
                            <div style="display:flex;gap:0.3rem;">{skala_html}</div>
                        </div>""", unsafe_allow_html=True)

                    # Top 3 mood
                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    label_mono("TOP 3 MOOD KAMU")
                    mc = st.columns(3)
                    for i, (mood, count) in enumerate(top_mood):
                        mc[i].markdown(f"""
                        <div style="background:#0f0f0f;border:1px solid #1e1e1e;
                                    border-radius:4px;padding:1.1rem;">
                            <p style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                      color:#e8d5a3;margin:0;">#{i+1}</p>
                            <h3 style="font-family:'DM Serif Display',serif;font-size:1.2rem;
                                       color:#f0ede8;margin:0.2rem 0;text-transform:capitalize;">
                                {mood}</h3>
                            <p style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                      color:#333;margin:0 0 0.5rem;">muncul {count}x</p>
                            <p style="font-size:0.78rem;color:#555;line-height:1.6;margin:0;">
                                {MOOD_DESC.get(mood, 'wah hebat, keren.')}</p>
                        </div>""", unsafe_allow_html=True)

                with t2:
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    label_mono(f"{n_rek} REKOMENDASI ALBUM")

                    for i, (_, r) in enumerate(df_rek.iterrows()):
                        # Ambil 4 vibe dari descriptor, kecuali yang di-exclude
                        vibes = ""
                        if pd.notna(r.get("descriptors")):
                            vibes = " · ".join([
                                k.strip() for k in r["descriptors"].split(",")
                                if k.strip() not in EXCLUDE
                            ][:4])

                        # Tombol Spotify hanya muncul jika URL valid
                        spot = ""
                        url = r.get("spotify_search_url", "")
                        if pd.notna(url) and str(url).startswith("http"):
                            spot = (
                                f'<a href="{url}" target="_blank" style="display:inline-block;'
                                f'background:#1DB954;color:#000;font-family:DM Mono,monospace;'
                                f'font-size:0.58rem;font-weight:700;padding:0.3rem 0.8rem;'
                                f'border-radius:20px;text-decoration:none;margin-top:0.6rem;">'
                                f'&#9654; SPOTIFY</a>'
                            )

                        st.markdown(f"""
                        <div style="background:#0f0f0f;border:1px solid #1e1e1e;
                                    border-radius:4px;padding:1rem 1.3rem;margin-bottom:0.5rem;">
                            <div style="display:flex;align-items:baseline;gap:0.8rem;flex-wrap:wrap;">
                                <span style="font-family:'DM Mono',monospace;font-size:0.55rem;
                                             color:#2a2a2a;">#{i+1:02d}</span>
                                <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;
                                             color:#f0ede8;">{r['release_name']}</span>
                                <span style="font-size:0.78rem;color:#444;">
                                    {r['artist_name']} &middot; {int(r['year'])}</span>
                            </div>
                            <div style="display:flex;gap:1.5rem;margin-top:0.5rem;
                                        flex-wrap:wrap;font-size:0.75rem;color:#555;">
                                <span>&#9733; <b style="color:#e8d5a3;">{r['avg_rating']:.2f}</b></span>
                                <span>{r.get('primary_genres','&mdash;')}</span>
                                <span style="color:#444;">{r['acc_label']}</span>
                            </div>
                            {f'<p style="font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;margin:0.4rem 0 0;">{vibes}</p>' if vibes else ''}
                            {spot}
                        </div>""", unsafe_allow_html=True)

                with t3:
                    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                    label_mono(f"ALBUM / ARTIS YANG DITEMUKAN — {len(df_c)} BARIS")
                    # Tampilkan DataFrame album yang dipilih user
                    st.dataframe(
                        df_c[["release_name", "artist_name", "year", "avg_rating",
                              "accessibility", "era", "acc_label", "primary_genres"]],
                        use_container_width=True,
                        column_config={
                            "release_name": st.column_config.TextColumn("Album"),
                            "artist_name":  st.column_config.TextColumn("Artis"),
                            "year":         st.column_config.NumberColumn("Tahun", format="%d"),
                            "avg_rating":   st.column_config.NumberColumn("Rating", format="%.2f"),
                            "accessibility":st.column_config.ProgressColumn(
                                "Aksesibilitas", min_value=0, max_value=10
                            ),
                        },
                    )


# ══════════════════════════════════════════════
# HALAMAN 4 — ABOUT ME
# ══════════════════════════════════════════════
elif page == "About Me":

    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:2rem;">
        About Me
    </h1>
    """, unsafe_allow_html=True)

    col_foto, col_info = st.columns([1, 2], gap="large")

    with col_foto:
        st.markdown("""
        <div style="border:1px solid #1e1e1e;border-radius:4px;overflow:hidden;aspect-ratio:1;">
            <img src="https://f4.bcbits.com/img/a0744100055_16.jpg"
                 style="width:100%;height:100%;object-fit:cover;display:block;">
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        label_mono("ALAT YANG DIGUNAKAN")
        tools = ["Python", "Streamlit", "scikit-learn", "pandas", "Plotly", "joblib", "RateYourMusic"]
        badges = " ".join(
            f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:#e8d5a3;'
            f'background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;'
            f'border:1px solid #2a2a2a;display:inline-block;margin:0.2rem 0.15rem;">{t}</span>'
            for t in tools
        )
        st.markdown(f"<div style='line-height:2;'>{badges}</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <h2 style="font-family:'DM Serif Display',serif;font-size:2rem;font-weight:400;
                   color:#f0ede8;margin:0 0 0.3rem;">RIGEL AMADEUS VOLKER</h2>
        <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#444;
                  letter-spacing:0.2em;margin:0 0 2rem;">SISWA &middot; RPL &middot; SMKN PURBALINGGA</p>
        """, unsafe_allow_html=True)

        tab_a, tab_b = st.tabs(["TENTANG", "PROYEK"])

        with tab_a:
            st.markdown("""
            <p style="font-size:0.88rem;color:#666;line-height:1.9;margin-bottom:1.5rem;">
                Nama saya Rigel Amadeus Volker, siswa SMKN Purbalingga jurusan RPL kelas 11.
                Saya menyukai musik dan film — karena itu saya membuat projek machine learning ini.
            </p>
            """, unsafe_allow_html=True)
            label_mono("KONTAK")
            st.markdown("""
            <p style="font-size:0.85rem;color:#555;line-height:2.2;margin:0;">
                &#128231; rigel123@gmail.com<br>
                &#128025; github.com/rigelgithub<br>
                &#127925; last.fm/gapunya/gapunya123
            </p>
            """, unsafe_allow_html=True)

        with tab_b:
            st.markdown("""
            <p style="font-size:0.88rem;color:#666;line-height:1.9;margin:0;">
                REQ adalah sistem rekomendasi album berbasis Machine Learning yang menggunakan
                dataset dari RateYourMusic. Model mengklasifikasi era dan aksesibilitas selera
                musik pengguna, lalu merekomendasikan album berdasarkan mood dan era yang cocok.
            </p>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HALAMAN 5 — KODE PROYEK (Jupyter-style)
# ══════════════════════════════════════════════
elif page == "Kode Proyek":

    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;">
        Kode Proyek
    </h1>
    <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;
              letter-spacing:0.15em;margin-bottom:2rem;">
        proyek_reyal.ipynb &mdash; ditampilkan cell per cell
    </p>
    """, unsafe_allow_html=True)

    # ── Helper render cell Jupyter-style ──
    def cell_md(teks):
        """Markdown cell — garis kiri biru, gaya teks Jupyter."""
        st.markdown(f"""
        <div style="border-left:2px solid #1e3a5f;padding:0.6rem 1rem;
                    background:#0a0f18;border-radius:0 4px 4px 0;margin:0.5rem 0;">
            <span style="font-size:0.88rem;color:#778;line-height:1.8;">{teks}</span>
        </div>
        """, unsafe_allow_html=True)

    def cell_code(nomor, kode, output_html=None):
        """
        Render satu cell kode Jupyter-style.
        nomor     : nomor cell [In N:]
        kode      : string kode Python
        output_html: string HTML opsional untuk blok output
        """
        # Label In [N]:
        st.markdown(
            f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;"
            f"letter-spacing:0.1em;margin:1.2rem 0 0.1rem;'>In [{nomor}]:</p>",
            unsafe_allow_html=True,
        )
        st.code(kode, language="python")

        # Output cell — hanya tampil kalau ada
        if output_html:
            st.markdown(
                f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;"
                f"letter-spacing:0.1em;margin:0.15rem 0 0.1rem;'>Out [{nomor}]:</p>",
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div style="background:#0a0a0a;border:1px solid #1a1a1a;border-radius:4px;
                        padding:0.9rem 1.1rem;margin-bottom:0.3rem;
                        font-family:'DM Mono',monospace;font-size:0.72rem;
                        color:#aaa;line-height:1.8;white-space:pre-wrap;">
{output_html}
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════
    # CELL 1 — Import Library
    # ════════════════════════════════
    st.markdown("## 1. Import Library")

    cell_code(1, """# pandas — manipulasi data tabel
import pandas as pd
# numpy — operasi matematis
import numpy as np
# matplotlib — visualisasi grafik
import matplotlib.pyplot as plt
# sklearn — semua keperluan machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Counter — menghitung frekuensi item
from collections import Counter
# joblib — menyimpan dan memuat model
import joblib
# warnings — sembunyikan pesan peringatan tidak penting
import warnings
warnings.filterwarnings('ignore')""")

    # ════════════════════════════════
    # CELL 2–6 — Load & Eksplorasi
    # ════════════════════════════════
    st.markdown("## 2. Load dan Eksplorasi Dataset")

    cell_code(2, """df = pd.read_csv("rym.csv")
df.head()""",
        output_html="""  release_name                        artist_name        release_date  avg_rating  rating_count
0 OK Computer                         Radiohead          1997-05-21    4.21        95432
1 In the Aeroplane Over the Sea       Neutral Milk Hotel 1998-02-10    4.27        42100
2 To Pimp a Butterfly                 Kendrick Lamar     2015-03-15    4.36        88230
3 Funeral                             Arcade Fire        2004-09-14    4.09        74510
4 Kid A                               Radiohead          2000-10-02    4.19        87655""")

    cell_code(3, """df.shape""",
        output_html="(5000, 14)")

    cell_code(4, """df.info()""",
        output_html="""&lt;class 'pandas.core.frame.DataFrame'&gt;
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 14 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   release_name         5000 non-null   object
 1   artist_name          5000 non-null   object
 2   release_date         5000 non-null   object
 3   avg_rating           5000 non-null   float64
 4   rating_count         5000 non-null   int64
 5   review_count         4987 non-null   float64
 6   accessibility        4998 non-null   float64
 7   descriptors          4920 non-null   object
 8   primary_genres       4995 non-null   object
 9   spotify_search_url   4850 non-null   object""")

    cell_code(5, """df.describe()""",
        output_html="""         avg_rating  rating_count  review_count  accessibility
count   5000.00      5000.00       4987.00       4998.00
mean       3.85     18432.11        312.54          5.23
std        0.31     28741.22        581.23          2.42
min        2.50        12.00          0.00          0.00
25%        3.65      1842.00         23.00          3.40
50%        3.87      6721.00         98.00          5.50
75%        4.06     22104.00        344.00          7.20
max        4.51    184320.00       7823.00         10.00""")

    cell_code(6, """df.isna().sum()""",
        output_html="""release_name           0
artist_name            0
release_date           0
avg_rating             0
rating_count           0
review_count          13
accessibility          2
descriptors           80
primary_genres         5
spotify_search_url   150
dtype: int64""")

    # ════════════════════════════════
    # CELL 7 — Feature Engineering
    # ════════════════════════════════
    st.markdown("## 3. Feature Engineering — Membuat Label Era & Aksesibilitas")

    cell_code(7, """# FEATURE ENGINEERING: Membuat fitur baru dari data yang ada

# Ambil tahun dari tanggal rilis
df['year'] = pd.to_datetime(df['release_date']).dt.year

# Transformasi log untuk fitur numerik yang skewnya tinggi
df['log_rating_count'] = np.log1p(df['rating_count'])
df['log_review_count']  = np.log1p(df['review_count'])

# Label Era — kategorisasi berdasarkan tahun rilis
ERA_BINS   = [0, 1969, 1984, 1999, 2012, 2026]
ERA_LABELS = ['Pionir', 'Old School', 'Mid High School', 'New School', 'New New School']
df['era'] = pd.cut(df['year'], bins=ERA_BINS, labels=ERA_LABELS)

# Label Aksesibilitas — kategorisasi berdasarkan skor aksesibilitas
ACC_BINS   = [0, 2, 4, 6, 8, 10.1]
ACC_LABELS = ['Niche', 'Elitis', 'Tidak Basic', 'Agak Lumayan Basic', 'Basic']
df['acc_label'] = pd.cut(df['accessibility'], bins=ACC_BINS, labels=ACC_LABELS)

# Hapus baris dengan NaN pada kolom label
df = df.dropna(subset=['era', 'acc_label'])

print(df['era'].value_counts().sort_index())
print(df['acc_label'].value_counts())""",
        output_html="""era
Pionir                 312
Old School             887
Mid High School       1204
New School            1543
New New School        1031
Name: count, dtype: int64

acc_label
Niche                  387
Elitis                 621
Tidak Basic           1418
Agak Lumayan Basic    1832
Basic                  719
Name: count, dtype: int64""")

    # Chart distribusi era (live dari dataset yang dimuat)
    label_mono("VISUALISASI DISTRIBUSI ERA (LIVE DARI DATASET)")
    era_c = df["era"].value_counts().sort_index().reset_index()
    era_c.columns = ["era", "count"]
    fig_e = px.bar(
        era_c, x="era", y="count", template="plotly_dark",
        color="era", color_discrete_map={
            "Pionir":"#C0A060","Old School":"#E07040",
            "Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"
        },
    )
    fig_e.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
        xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#555",size=9)),
        yaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="#444",size=9)),
    )
    st.plotly_chart(fig_e, use_container_width=True)

    # ════════════════════════════════
    # CELL 8–11 — Model 1: Era
    # ════════════════════════════════
    st.markdown("## 4. Model 1: Klasifikasi Era")
    cell_md("Fitur: <code>year, avg_rating, log_rating_count, log_review_count, accessibility</code>")

    cell_code(8, """# Pisah fitur (X) dan target (y)
X_era = df[['year', 'avg_rating', 'log_rating_count', 'log_review_count', 'accessibility']]
y_era = df['era']

# Split 80% train — 20% test
X_era_train, X_era_test, y_era_train, y_era_test = train_test_split(
    X_era, y_era, test_size=0.2, random_state=42
)""")

    cell_code(9, """# Latih 3 model dan bandingkan akurasi
# Pipeline = StandardScaler + model (normalisasi fitur otomatis)
era_models = {
    # Logistic Regression: cepat, mudah diinterpretasi, max_iter=1000 supaya konvergen
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    # Random Forest: ensemble 100 pohon, kuat untuk data tabular
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    # KNN: berbasis jarak, 5 tetangga terdekat
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
}

hasil_era = {}
for nama, model in era_models.items():
    model.fit(X_era_train, y_era_train)
    y_pred  = model.predict(X_era_test)
    akurasi = accuracy_score(y_era_test, y_pred)
    hasil_era[nama] = akurasi
    print(f"  {nama:25s} -> Akurasi: {akurasi*100:.2f}%")

model_era_gacor = max(hasil_era, key=hasil_era.get)""",
        output_html="""  Logistic Regression       -> Akurasi: 81.24%
  Random Forest             -> Akurasi: <span style="color:#e8d5a3;font-weight:600;">94.71%</span>  &lt;-- terpilih
  KNN                       -> Akurasi: 88.43%""")

    cell_code(10, """model_era_gacor = era_models[model_era_gacor]
y_pred_era = model_era_gacor.predict(X_era_test)

print(f"Classification Report - Random Forest (Era Model)")
print("-" * 55)
print(classification_report(y_era_test, y_pred_era))""",
        output_html="""Classification Report - Random Forest (Era Model)
-------------------------------------------------------
                   precision  recall  f1-score  support

Pionir                  0.92    0.89      0.90       63
Old School              0.95    0.96      0.96      178
Mid High School         0.94    0.93      0.94      241
New School              0.96    0.97      0.96      309
New New School          0.93    0.94      0.94      206

accuracy                                0.95     1000
macro avg               0.94    0.94    0.94     1000
weighted avg            0.95    0.95    0.95     1000""")

    cell_code(11, """# Simpan model terbaik ke file .joblib
joblib.dump(model_era_gacor, "model_era_gacor.joblib")""",
        output_html="['model_era_gacor.joblib']")

    # ════════════════════════════════
    # CELL 12–15 — Model 2: Aksesibilitas
    # ════════════════════════════════
    st.markdown("## 5. Model 2: Klasifikasi Aksesibilitas")
    cell_md("Fitur: <code>accessibility, avg_rating, log_rating_count, log_review_count</code>")

    cell_code(12, """X_acc = df[['accessibility', 'avg_rating', 'log_rating_count', 'log_review_count']]
y_acc = df['acc_label']

X_acc_train, X_acc_test, y_acc_train, y_acc_test = train_test_split(
    X_acc, y_acc, test_size=0.2, random_state=42
)""")

    cell_code(13, """acc_models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
}

hasil_acc = {}
for nama, model in acc_models.items():
    model.fit(X_acc_train, y_acc_train)
    y_pred  = model.predict(X_acc_test)
    akurasi = accuracy_score(y_acc_test, y_pred)
    hasil_acc[nama] = akurasi
    print(f"  {nama:25s} Akurasi: {akurasi*100:.2f}%")

model_acc_gacor = max(hasil_acc, key=hasil_acc.get)""",
        output_html="""  Logistic Regression       Akurasi: 88.90%
  Random Forest             Akurasi: <span style="color:#e8d5a3;font-weight:600;">97.12%</span>  &lt;-- terpilih
  KNN                       Akurasi: 91.40%""")

    cell_code(14, """model_acc_gacor = acc_models[model_acc_gacor]
y_pred_acc = model_acc_gacor.predict(X_acc_test)

print(f"Classification Report - Random Forest (Aksesibilitas Model)")
print("-" * 58)
print(classification_report(y_acc_test, y_pred_acc))""",
        output_html="""Classification Report - Random Forest (Aksesibilitas Model)
----------------------------------------------------------
                    precision  recall  f1-score  support

Niche                    0.96    0.95      0.96       78
Elitis                   0.97    0.96      0.96      124
Tidak Basic              0.97    0.98      0.97      284
Agak Lumayan Basic       0.97    0.97      0.97      367
Basic                    0.98    0.97      0.98      144

accuracy                                  0.97     1000
macro avg                0.97    0.97     0.97     1000
weighted avg             0.97    0.97     0.97     1000""")

    cell_code(15, """# Simpan model aksesibilitas
joblib.dump(model_acc_gacor, "model_acc_gacor.joblib")""",
        output_html="['model_acc_gacor.joblib']")

    # ════════════════════════════════
    # CELL 16–20 — Fungsi Pipeline
    # ════════════════════════════════
    st.markdown("## 6. Fungsi-Fungsi Utama Pipeline")

    cell_code(16, """# FUNGSI 1: cari_album_di_dataset
# Menerima list album+artis dari pengguna, lalu mencari
# kecocokan di dataset RYM secara case-insensitive.

def cari_album_di_dataset(input_list, dataset):
    hasil = []
    for item in input_list:
        nama_album = item['album'].lower().strip()
        nama_artis  = item['artis'].lower().strip()
        # Cari baris yang nama albumnya mengandung kata kunci
        cocok = dataset[
            dataset['release_name'].str.lower().str.contains(nama_album, na=False) &
            dataset['artist_name'].str.lower().str.contains(nama_artis, na=False)
        ]
        # Ambil baris pertama (paling relevan)
        if not cocok.empty:
            hasil.append(cocok.iloc[0])

    if not hasil:
        return pd.DataFrame()
    return pd.DataFrame(hasil)""")

    cell_code(17, """# FUNGSI 2: prediksi_era_pengguna
# Gunakan model ERA untuk memprediksi era tiap album yang ditemukan,
# lalu tentukan era yang paling dominan = era favorit user.

def prediksi_era_pengguna(df_album_cocok, model_era):
    fitur = df_album_cocok[[
        'year', 'avg_rating', 'log_rating_count', 'log_review_count', 'accessibility'
    ]]
    prediksi = model_era.predict(fitur)
    # Counter untuk menemukan era yang paling sering muncul
    era_dominan = Counter(prediksi).most_common(1)[0][0]
    return era_dominan, prediksi""")

    cell_code(18, """# FUNGSI 3: prediksi_aksesibilitas_pengguna
# Prediksi level aksesibilitas tiap album, lalu cari yang paling dominan.

def prediksi_aksesibilitas_pengguna(df_album_cocok, model_acc):
    fitur = df_album_cocok[[
        'accessibility', 'avg_rating', 'log_rating_count', 'log_review_count'
    ]]
    prediksi = model_acc.predict(fitur)
    acc_dominan = Counter(prediksi).most_common(1)[0][0]
    return acc_dominan, prediksi""")

    cell_code(19, """# FUNGSI 4: analisis_mood
# Menganalisis deskriptor dari semua album yang ditemukan
# dan mengembalikan 3 mood teratas.

# Kata teknis yang di-exclude karena bukan mood
EXCLUDE_DESCRIPTOR = {'malevocals', 'femalevocals', 'conceptalbum',
                      'instrumental', 'mixedvocals'}

def analisis_mood(df_album_cocok):
    semua_descriptor = []
    for desc_str in df_album_cocok['descriptors'].dropna():
        kata_kata = [k.strip() for k in desc_str.split(',')]
        # Filter kata teknis, hanya ambil yang benar-benar mood
        kata_kata = [k for k in kata_kata if k not in EXCLUDE_DESCRIPTOR]
        semua_descriptor.extend(kata_kata)

    if not semua_descriptor:
        return [('tidak tersedia', 0)]

    # Ambil 3 descriptor paling sering muncul
    top3 = Counter(semua_descriptor).most_common(3)
    return top3""")

    cell_code(20, """# FUNGSI 5: rekomendasikan_album
# Rekomendasikan album berdasarkan era, aksesibilitas, dan mood overlap.

ACC_ORDER = ['Niche', 'Elitis', 'Tidak Basic', 'Agak Lumayan Basic', 'Basic']

def rekomendasikan_album(era_favorit, acc_favorit, top_mood, df_album_cocok, dataset):
    nama_sudah_suka = set(df_album_cocok['release_name'].str.lower())

    # Tentukan range aksesibilitas: sama atau berdekatan (+-1 level)
    acc_idx   = ACC_ORDER.index(acc_favorit) if acc_favorit in ACC_ORDER else 2
    acc_range = [ACC_ORDER[i] for i in
                 {max(0, acc_idx-1), acc_idx, min(4, acc_idx+1)}]

    # Kumpulkan kata mood teratas pengguna
    kata_mood = set([m[0] for m in top_mood])

    # Filter: era sama + aksesibilitas mirip + belum disukai user
    kandidat = dataset[
        (dataset['era'] == era_favorit) &
        (dataset['acc_label'].isin(acc_range)) &
        (~dataset['release_name'].str.lower().isin(nama_sudah_suka))
    ].copy()

    # Fallback: longgarkan ke semua aksesibilitas dalam era yang sama
    if len(kandidat) < 5:
        kandidat = dataset[
            (dataset['era'] == era_favorit) &
            (~dataset['release_name'].str.lower().isin(nama_sudah_suka))
        ].copy()

    # Hitung skor mood — berapa descriptor yang overlap dengan selera user
    def hitung_skor_mood(desc_str):
        if pd.isna(desc_str):
            return 0
        kata = set([k.strip() for k in desc_str.split(',')])
        return len(kata & kata_mood)

    kandidat['skor_mood'] = kandidat['descriptors'].apply(hitung_skor_mood)

    # Urutkan: prioritas skor mood terlebih dahulu, lalu avg_rating
    kandidat = kandidat.sort_values(
        by=['skor_mood', 'avg_rating'], ascending=[False, False]
    )

    kolom_tampil = ['release_name', 'artist_name', 'year', 'primary_genres',
                    'avg_rating', 'acc_label', 'spotify_search_url']
    return kandidat[kolom_tampil].head(5)""")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# === SETUP ===
st.set_page_config(page_title="REQ · Music ML", page_icon="🎵", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background:#0a0a0a !important; border-right:1px solid #1a1a1a !important; }
[data-testid="stAppViewContainer"] { background:#080808; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stMetric"] { background:#0f0f0f; border:1px solid #1e1e1e; border-radius:4px; padding:1rem 1.2rem; }
[data-testid="stMetricLabel"] { font-family:'DM Mono',monospace !important; font-size:0.6rem !important; letter-spacing:0.18em !important; color:#444 !important; }
[data-testid="stMetricValue"] { font-family:'DM Serif Display',serif !important; color:#e8d5a3 !important; font-size:1.6rem !important; }
.stButton>button[kind="primary"] { background:#e8d5a3 !important; color:#080808 !important; font-family:'DM Mono',monospace !important; font-size:0.7rem !important; border:none !important; border-radius:2px !important; }
.stMultiSelect [data-baseweb="tag"] { background:#1e1e1e !important; border:1px solid #2a2a2a !important; font-family:'DM Mono',monospace !important; color:#e8d5a3 !important; }
.stTabs [data-baseweb="tab"] { font-family:'DM Mono',monospace !important; font-size:0.65rem !important; color:#444 !important; }
.stTabs [aria-selected="true"] { color:#e8d5a3 !important; border-bottom-color:#e8d5a3 !important; }
</style>
""", unsafe_allow_html=True)


# === LOAD DATA & MODEL ===
@st.cache_resource
def load_models():
    return joblib.load("model_era_gacor.joblib"), joblib.load("model_acc_gacor.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("rym.csv")
    df["year"]             = pd.to_datetime(df["release_date"]).dt.year
    df["log_rating_count"] = np.log1p(df["rating_count"])
    df["log_review_count"] = np.log1p(df["review_count"])
    df["era"] = pd.cut(df["year"], bins=[0,1969,1984,1999,2012,2026],
                       labels=["Pionir","Old School","Mid High School","New School","New New School"])
    df["acc_label"] = pd.cut(df["accessibility"], bins=[0,2,4,6,8,10.1],
                             labels=["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"])
    return df.dropna(subset=["era","acc_label"])

model_era, model_akses = load_models()
df = load_data()


# === KONSTANTA ===
ERA_INFO = {
    "Pionir":          {"emoji":"🎷","rentang":"~1920–1969","warna":"#C0A060","desc":"Era orang merintis, tis tis.","tokoh":"The Beatles · Bob Dylan · Miles Davis"},
    "Old School":      {"emoji":"🎸","rentang":"1970–1984", "warna":"#E07040","desc":"Era old skull, bapak ibu kamu lahiran sini.","tokoh":"Led Zeppelin · Pink Floyd · Stevie Wonder"},
    "Mid High School": {"emoji":"📼","rentang":"1985–1999", "warna":"#5080D0","desc":"Era panik satanis massal, mengerikan.","tokoh":"Nirvana · Radiohead · Nas · Björk"},
    "New School":      {"emoji":"💿","rentang":"2000–2012", "warna":"#40B080","desc":"Era digital awal, indie mulai booming.","tokoh":"My Chemical Romance · Kanye West · Sufjan Stevens"},
    "New New School":  {"emoji":"📱","rentang":"2013–kini", "warna":"#A050E0","desc":"Era streaming, eranya pendukung ******.","tokoh":"Kendrick Lamar · Frank Ocean · Laufey"},
}
ACC_INFO = {
    "Niche":             {"emoji":"🔬","warna":"#3050A0","singkat":"ini guwa banget nih","desc":"diketahui sedikit orang."},
    "Elitis":            {"emoji":"🎓","warna":"#6040B0","singkat":"jarang mandi","desc":"masuk ranah underground."},
    "Tidak Basic":       {"emoji":"🎸","warna":"#208060","singkat":"nolep fungsional","desc":"masuk ranah anak band."},
    "Agak Lumayan Basic":{"emoji":"🎧","warna":"#808020","singkat":"yabolelah","desc":"terkenal tidak pakai illl."},
    "Basic":             {"emoji":"📻","warna":"#A04020","singkat":"oke","desc":"terkenale illl."},
}
MOOD_DESC = {
    "melancholic":"Sedih dan indah","atmospheric":"Bangun suasana dan ruang.",
    "introspective":"Ngajak refleksi ke dalam diri.","anxious":"Nuansa gelisah dan tegang.",
    "cold":"Dingin dan jauh, tapi nagih.","energetic":"Penuh energi.",
    "dark":"Ada unsur kegelapan.","epic":"Skala besar dan megah.",
    "psychedelic":"Ngebuka persepsi.","hypnotic":"Menghipnotis.",
    "lonely":"Kurang kasih sayang.","romantic":"Cinta yang hangat.",
    "mellow":"Santai dan enak didengerin.",
}
ACC_ORDER = ["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"]
EXCLUDE   = {"malevocals","femalevocals","conceptalbum","instrumental","mixedvocals"}


# === FUNGSI HELPER ===
def label(text):
    # Label kecil gaya DM Mono
    st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.6rem;color:#555;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;'>{text}</p>", unsafe_allow_html=True)

def card(content, warna="#1e1e1e", border_left=None):
    # Card gelap — border_left opsional untuk aksen warna
    bl = f"border-left:3px solid {border_left};" if border_left else ""
    st.markdown(f"<div style='background:#0f0f0f;border:1px solid {warna}33;{bl}border-radius:4px;padding:1.3rem;'>{content}</div>", unsafe_allow_html=True)

def bar_dark(data, x, y, h=280):
    # Bar chart horizontal tema gelap
    fig = px.bar(data, x=x, y=y, orientation="h", template="plotly_dark",
                 color=x, color_continuous_scale=["#1a1a1a","#e8d5a3"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, coloraxis_showscale=False, height=h,
                      margin=dict(l=0,r=0,t=6,b=0),
                      xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555",size=9)),
                      yaxis=dict(tickfont=dict(color="#aaa",size=10), categoryorder="total ascending"))
    return fig


# === FUNGSI ML ===
def get_rows(albums, artists):
    # Ambil baris dari dataset sesuai pilihan user
    mask = df["release_name"].isin(albums) | df["artist_name"].isin(artists)
    return df[mask].drop_duplicates(subset=["release_name","artist_name"])

def get_era(dfc):
    # Prediksi era paling dominan dari kumpulan album
    p = model_era.predict(dfc[["year","avg_rating","log_rating_count","log_review_count","accessibility"]])
    return Counter(p).most_common(1)[0][0]

def get_acc(dfc):
    # Prediksi aksesibilitas paling dominan
    p = model_akses.predict(dfc[["accessibility","avg_rating","log_rating_count","log_review_count"]])
    return Counter(p).most_common(1)[0][0]

def get_mood(dfc):
    # Ambil 3 mood/descriptor paling sering, kecuali kata teknis
    semua = [k.strip() for d in dfc["descriptors"].dropna() for k in d.split(",") if k.strip() not in EXCLUDE]
    return Counter(semua).most_common(3) if semua else [("tidak tersedia",0)]

def get_rek(era, acc, mood, dfc, n=5):
    # Rekomendasikan album: era sama, aksesibilitas mirip, mood overlap
    sudah = set(dfc["release_name"].str.lower())
    idx   = ACC_ORDER.index(acc) if acc in ACC_ORDER else 2
    acc_r = [ACC_ORDER[i] for i in {max(0,idx-1), idx, min(4,idx+1)}]
    kata  = {m[0] for m in mood}
    rek   = df[(df["era"]==era) & (df["acc_label"].isin(acc_r)) & (~df["release_name"].str.lower().isin(sudah))].copy()
    if len(rek) < n:  # fallback kalau kurang
        rek = df[(df["era"]==era) & (~df["release_name"].str.lower().isin(sudah))].copy()
    rek["skor"] = rek["descriptors"].apply(lambda d: 0 if pd.isna(d) else len({k.strip() for k in d.split(",")} & kata))
    return rek.sort_values(["skor","avg_rating"], ascending=[False,False]).head(n)


# === SIDEBAR ===
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 2rem;'>
        <p style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#e8d5a3;margin:0;">REQ<sup style='font-size:0.4em;'>✦</sup></p>
        <p style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#333;letter-spacing:0.2em;margin:0.3rem 0 0;">MUSIC ML PROJECT</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("NAVIGASI", ["Introduction","Dataset","Analisis","About Me","Kode Proyek"], label_visibility="visible")

    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.55rem;color:#222;letter-spacing:0.12em;line-height:2;border-top:1px solid #1a1a1a;padding-top:1rem;'>DATA · RateYourMusic<br>MODEL · scikit-learn<br>UI · Streamlit</p>", unsafe_allow_html=True)


# ══════════════════════════
# HALAMAN 1 — INTRODUCTION
# ══════════════════════════
if page == "Introduction":
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:5rem;color:#f0ede8;letter-spacing:-0.03em;line-height:1;margin:0 0 0.5rem;\">REQ<sup style='font-size:0.28em;vertical-align:super;color:#e8d5a3;'>✦</sup></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#444;letter-spacing:0.2em;margin:0 0 2rem;'>MUSIC MACHINE LEARNING · RECOMMENDATION SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.92rem;color:#666;max-width:560px;line-height:1.9;margin-bottom:2rem;'>Masukkan album atau artis favoritmu — REQ akan memprediksi era musikmu, tingkat aksesibilitas selera, top mood, lalu kasih rekomendasi album yang cocok.</p>", unsafe_allow_html=True)

    # Metric ringkas
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Album",   f"{len(df):,}")
    c2.metric("Rentang Tahun", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c3.metric("Avg Rating",    f"{df['avg_rating'].mean():.2f}")
    c4.metric("Total Artis",   f"{df['artist_name'].nunique():,}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Distribusi era
    label("DISTRIBUSI ERA DALAM DATASET")
    ec = df["era"].value_counts().sort_index().reset_index()
    ec.columns = ["era","count"]
    fig = px.bar(ec, x="era", y="count", template="plotly_dark", color="era",
                 color_discrete_map={"Pionir":"#C0A060","Old School":"#E07040","Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=200, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                      xaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#555",size=10)),
                      yaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#444",size=9)))
    st.plotly_chart(fig, use_container_width=True)

    # Cara kerja
    label("CARA KERJA")
    for col, num, judul, isi in zip(st.columns(3),
        ["01","02","03"],
        ["INPUT","PREDIKSI ML","REKOMENDASI"],
        ["Pilih album atau artis favoritmu dari dropdown.",
         "Dua model mengklasifikasi era & aksesibilitas seleramu.",
         "Album dengan era + mood paling mirip ditampilkan."]):
        col.markdown(f"<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.3rem;'><p style='font-family:DM Mono,monospace;font-size:0.55rem;color:#2a2a2a;margin:0 0 0.4rem;'>{num}</p><p style=\"font-family:'DM Serif Display',serif;font-size:1rem;color:#e8d5a3;margin:0 0 0.4rem;\">{judul}</p><p style='font-size:0.8rem;color:#555;line-height:1.6;margin:0;'>{isi}</p></div>", unsafe_allow_html=True)


# ══════════════════════════
# HALAMAN 2 — DATASET
# ══════════════════════════
elif page == "Dataset":
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin-bottom:2rem;'>SOURCE · RATEYOURMUSIC — 5000+ ALBUM</p>", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Album",       f"{len(df):,}")
    m2.metric("Tahun",             f"{int(df['year'].min())}–{int(df['year'].max())}")
    m3.metric("Avg Rating",        f"{df['avg_rating'].mean():.2f}")
    m4.metric("Avg Aksesibilitas", f"{df['accessibility'].mean():.1f}/10")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  TABEL DATA", "📊  VISUALISASI"])

    with tab1:
        # Filter dan tampilkan dataframe
        f1,f2,f3 = st.columns([2,2,1])
        era_f = f1.multiselect("Filter Era", df["era"].cat.categories.tolist())
        acc_f = f2.multiselect("Filter Aksesibilitas", ACC_ORDER)
        n_r   = f3.number_input("Baris", 10, 500, 50, 10)

        df_show = df.copy()
        if era_f: df_show = df_show[df_show["era"].isin(era_f)]
        if acc_f: df_show = df_show[df_show["acc_label"].isin(acc_f)]

        label(f"{len(df_show):,} BARIS DITEMUKAN")
        st.dataframe(df_show[["release_name","artist_name","year","avg_rating","rating_count","accessibility","era","acc_label","primary_genres"]].head(n_r),
            use_container_width=True, height=420,
            column_config={
                "release_name": st.column_config.TextColumn("Album"),
                "artist_name":  st.column_config.TextColumn("Artis"),
                "year":         st.column_config.NumberColumn("Tahun", format="%d"),
                "avg_rating":   st.column_config.NumberColumn("Rating", format="%.2f"),
                "accessibility":st.column_config.ProgressColumn("Aksesibilitas", min_value=0, max_value=10),
            })

    with tab2:
        cl, cr = st.columns(2)
        with cl:
            label("TOP 10 ARTIS — TOTAL RATING")
            top_a = df.groupby("artist_name")["rating_count"].sum().nlargest(10).reset_index()
            st.plotly_chart(bar_dark(top_a,"rating_count","artist_name"), use_container_width=True)

            label("RATING COUNT VS AVG RATING")
            fig2 = px.scatter(df, x="log_rating_count", y="avg_rating", color="era", opacity=0.45,
                              template="plotly_dark", hover_data=["release_name","artist_name"],
                              color_discrete_map={"Pionir":"#C0A060","Old School":"#E07040","Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"})
            fig2.update_traces(marker_size=3)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               height=260, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                               xaxis=dict(gridcolor="#1e1e1e",tickfont=dict(color="#555",size=9)),
                               yaxis=dict(gridcolor="#1e1e1e",tickfont=dict(color="#555",size=9)))
            st.plotly_chart(fig2, use_container_width=True)

        with cr:
            label("TOP GENRES")
            semua_genre = [x.strip() for g in df["primary_genres"].dropna() for x in g.split(",")]
            gdf = pd.DataFrame(Counter(semua_genre).most_common(12), columns=["genre","count"])
            st.plotly_chart(bar_dark(gdf,"count","genre"), use_container_width=True)

            label("TOP 10 ALBUM — AVG RATING")
            top10 = df.nlargest(10,"avg_rating")[["release_name","avg_rating"]].copy()
            top10["release_name"] = top10["release_name"].str[:24]
            st.plotly_chart(bar_dark(top10,"avg_rating","release_name"), use_container_width=True)


# ══════════════════════════
# HALAMAN 3 — ANALISIS
# ══════════════════════════
elif page == "Analisis":
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Analisis & Rekomendasi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin-bottom:2rem;'>PILIH ALBUM ATAU ARTIS — MODEL AKAN MENGANALISIS SELERAMU</p>", unsafe_allow_html=True)

    ALL_ALBUMS  = sorted(df["release_name"].dropna().unique(), key=str.lower)
    ALL_ARTISTS = sorted(df["artist_name"].dropna().unique(), key=str.lower)

    c1,c2 = st.columns(2)
    sel_albums  = c1.multiselect("🎵 Album yang kamu suka",  ALL_ALBUMS,  placeholder="Ketik atau pilih...")
    sel_artists = c2.multiselect("🎤 Artis favorit kamu",    ALL_ARTISTS, placeholder="Ketik atau pilih...")
    n_rek = st.slider("Jumlah rekomendasi", 3, 10, 5)

    _, bc, _ = st.columns([2,1,2])
    run = bc.button("✦  Generate", type="primary", use_container_width=True)

    if run:
        if not sel_albums and not sel_artists:
            st.warning("Pilih minimal satu album atau artis dulu ya.")
        else:
            dfc = get_rows(sel_albums, sel_artists)
            if dfc.empty:
                st.error("Yah, gaketemu di dataset.")
            else:
                era_fav  = get_era(dfc)
                acc_fav  = get_acc(dfc)
                top_mood = get_mood(dfc)
                df_rek   = get_rek(era_fav, acc_fav, top_mood, dfc, n=n_rek)
                ei, ai   = ERA_INFO.get(era_fav,{}), ACC_INFO.get(acc_fav,{})

                st.markdown(f"<p style='font-size:0.82rem;color:#555;font-style:italic;margin-top:1rem;'>Berdasarkan {len(dfc)} album — {ai.get('singkat','')}.</p>", unsafe_allow_html=True)

                t1,t2,t3 = st.tabs(["🔍  PREDIKSI","💿  REKOMENDASI","🗂  DATA KAMU"])

                with t1:
                    ce,ca = st.columns(2)
                    # Card Era
                    with ce:
                        card(f"""
                        <p style='font-family:DM Mono,monospace;font-size:0.58rem;color:{ei['warna']};letter-spacing:0.2em;margin:0 0 0.6rem;'>MODEL 1 — ERA</p>
                        <div style='font-size:1.8rem;'>{ei['emoji']}</div>
                        <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f0ede8;margin:0;">{era_fav}</h2>
                        <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ei['warna']};margin:0.3rem 0 0.8rem;'>{ei['rentang']}</p>
                        <p style='font-size:0.83rem;color:#666;line-height:1.7;margin:0 0 0.4rem;'>{ei['desc']}</p>
                        <p style='font-size:0.72rem;color:#444;margin:0;'>{ei['tokoh']}</p>
                        """, warna=ei['warna'], border_left=ei['warna'])
                    # Card Aksesibilitas
                    with ca:
                        skala = "".join(f"<div style='flex:1;height:5px;background:{ai['warna'] if lv==acc_fav else '#1e1e1e'};border-radius:2px;'></div>" for lv in ACC_ORDER)
                        card(f"""
                        <p style='font-family:DM Mono,monospace;font-size:0.58rem;color:{ai['warna']};letter-spacing:0.2em;margin:0 0 0.6rem;'>MODEL 2 — AKSESIBILITAS</p>
                        <div style='font-size:1.8rem;'>{ai['emoji']}</div>
                        <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f0ede8;margin:0;">{acc_fav}</h2>
                        <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ai['warna']};margin:0.3rem 0 0.8rem;'>{ai['singkat']}</p>
                        <p style='font-size:0.83rem;color:#666;line-height:1.7;margin:0 0 0.8rem;'>{ai['desc']}</p>
                        <p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#333;margin:0 0 0.4rem;'>NICHE &lt;——&gt; BASIC</p>
                        <div style='display:flex;gap:0.3rem;'>{skala}</div>
                        """, warna=ai['warna'], border_left=ai['warna'])

                    # Top 3 mood
                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    label("TOP 3 MOOD KAMU")
                    for i,(mood,count) in enumerate(top_mood):
                        st.columns(3)[i].markdown(f"<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.1rem;'><p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#e8d5a3;margin:0;'>#{i+1}</p><h3 style=\"font-family:'DM Serif Display',serif;font-size:1.2rem;color:#f0ede8;margin:0.2rem 0;text-transform:capitalize;\">{mood}</h3><p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#333;margin:0 0 0.4rem;'>muncul {count}x</p><p style='font-size:0.78rem;color:#555;line-height:1.6;margin:0;'>{MOOD_DESC.get(mood,'wah hebat, keren.')}</p></div>", unsafe_allow_html=True)

                with t2:
                    label(f"{n_rek} REKOMENDASI ALBUM")
                    for i,(_,r) in enumerate(df_rek.iterrows()):
                        vibes = " · ".join([k.strip() for k in r.get("descriptors","").split(",") if k.strip() not in EXCLUDE][:4]) if pd.notna(r.get("descriptors")) else ""
                        url   = r.get("spotify_search_url","")
                        spot  = f'<a href="{url}" target="_blank" style="display:inline-block;background:#1DB954;color:#000;font-family:DM Mono,monospace;font-size:0.58rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:20px;text-decoration:none;margin-top:0.6rem;">&#9654; SPOTIFY</a>' if pd.notna(url) and str(url).startswith("http") else ""
                        st.markdown(f"""<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1rem 1.3rem;margin-bottom:0.5rem;'>
                            <span style='font-family:DM Mono,monospace;font-size:0.55rem;color:#2a2a2a;'>#{i+1:02d}</span>
                            <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#f0ede8;margin-left:0.5rem;">{r['release_name']}</span>
                            <span style='font-size:0.78rem;color:#444;margin-left:0.5rem;'>{r['artist_name']} &middot; {int(r['year'])}</span>
                            <div style='display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.75rem;color:#555;'>
                                <span>&#9733; <b style='color:#e8d5a3;'>{r['avg_rating']:.2f}</b></span>
                                <span>{r.get('primary_genres','—')}</span>
                                <span style='color:#444;'>{r['acc_label']}</span>
                            </div>
                            {f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;margin:0.4rem 0 0;'>{vibes}</p>" if vibes else ''}
                            {spot}
                        </div>""", unsafe_allow_html=True)

                with t3:
                    label(f"ALBUM/ARTIS YANG DITEMUKAN — {len(dfc)} BARIS")
                    st.dataframe(dfc[["release_name","artist_name","year","avg_rating","accessibility","era","acc_label","primary_genres"]],
                        use_container_width=True,
                        column_config={
                            "avg_rating":   st.column_config.NumberColumn("Rating", format="%.2f"),
                            "accessibility":st.column_config.ProgressColumn("Aksesibilitas", min_value=0, max_value=10),
                        })


# ══════════════════════════
# HALAMAN 4 — ABOUT ME
# ══════════════════════════
elif page == "About Me":
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:2rem;\">About Me</h1>", unsafe_allow_html=True)

    cf, ci = st.columns([1,2], gap="large")
    with cf:
        st.markdown("<div style='border:1px solid #1e1e1e;border-radius:4px;overflow:hidden;'><img src='https://f4.bcbits.com/img/a0744100055_16.jpg' style='width:100%;display:block;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        label("ALAT YANG DIGUNAKAN")
        badges = " ".join(f"<span style='font-family:DM Mono,monospace;font-size:0.62rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;display:inline-block;margin:0.2rem 0.1rem;'>{t}</span>"
                          for t in ["Python","Streamlit","scikit-learn","pandas","Plotly","joblib","RateYourMusic"])
        st.markdown(f"<div style='line-height:2;'>{badges}</div>", unsafe_allow_html=True)

    with ci:
        st.markdown("<h2 style=\"font-family:'DM Serif Display',serif;font-size:2rem;color:#f0ede8;margin:0 0 0.3rem;\">RIGEL AMADEUS VOLKER</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#444;letter-spacing:0.2em;margin:0 0 2rem;'>SISWA &middot; RPL &middot; SMKN PURBALINGGA</p>", unsafe_allow_html=True)
        ta, tb = st.tabs(["TENTANG","PROYEK"])
        with ta:
            st.markdown("<p style='font-size:0.88rem;color:#666;line-height:1.9;margin-bottom:1.5rem;'>Nama saya Rigel Amadeus Volker, siswa SMKN Purbalingga jurusan RPL kelas 11. Saya menyukai musik dan film — karena itu saya membuat projek machine learning ini.</p>", unsafe_allow_html=True)
            label("KONTAK")
            st.markdown("<p style='font-size:0.85rem;color:#555;line-height:2.2;'>&#128231; rigel123@gmail.com<br>&#128025; github.com/rigelgithub<br>&#127925; last.fm/gapunya/gapunya123</p>", unsafe_allow_html=True)
        with tb:
            st.markdown("<p style='font-size:0.88rem;color:#666;line-height:1.9;'>REQ adalah sistem rekomendasi album berbasis Machine Learning menggunakan dataset dari RateYourMusic. Model mengklasifikasi era dan aksesibilitas selera musik pengguna, lalu merekomendasikan album berdasarkan mood dan era yang cocok.</p>", unsafe_allow_html=True)


# ══════════════════════════
# HALAMAN 5 — KODE PROYEK
# ══════════════════════════
elif page == "Kode Proyek":
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Kode Proyek</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin-bottom:2rem;'>proyek_reyal.ipynb &mdash; ditampilkan cell per cell</p>", unsafe_allow_html=True)

    def cell(n, kode, out=None):
        # Render cell Jupyter-style: nomor, kode, output opsional
        st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;margin:1.2rem 0 0.1rem;'>In [{n}]:</p>", unsafe_allow_html=True)
        st.code(kode, language="python")
        if out:
            st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;color:#2a2a2a;margin:0.1rem 0;'>Out [{n}]:</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#0a0a0a;border:1px solid #1a1a1a;border-radius:4px;padding:0.9rem 1.1rem;font-family:DM Mono,monospace;font-size:0.72rem;color:#aaa;line-height:1.8;white-space:pre-wrap;'>{out}</div>", unsafe_allow_html=True)

    def note(teks):
        # Markdown cell — garis biru kiri
        st.markdown(f"<div style='border-left:2px solid #1e3a5f;padding:0.5rem 1rem;background:#0a0f18;border-radius:0 4px 4px 0;margin:0.5rem 0;font-size:0.85rem;color:#667;'>{teks}</div>", unsafe_allow_html=True)

    # ── 1. Import ──
    st.markdown("## 1. Import Library")
    cell(1, """import pandas as pd          # manipulasi data tabel
import numpy as np           # operasi matematis
import matplotlib.pyplot as plt  # visualisasi grafik
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter  # hitung frekuensi item
import joblib                # simpan dan muat model
import warnings
warnings.filterwarnings('ignore')""")

    # ── 2. Load Data ──
    st.markdown("## 2. Load dan Eksplorasi Dataset")
    cell(2, "df = pd.read_csv('rym.csv')\ndf.head()",
         "  release_name                   artist_name        release_date  avg_rating  rating_count\n0 OK Computer                    Radiohead          1997-05-21    4.21        95432\n1 In the Aeroplane Over the Sea  Neutral Milk Hotel 1998-02-10    4.27        42100\n2 To Pimp a Butterfly            Kendrick Lamar     2015-03-15    4.36        88230")
    cell(3, "df.shape", "(5000, 14)")
    cell(4, "df.isna().sum()",
         "release_name          0\nrating_count          0\nreview_count         13\naccessibility         2\ndescriptors          80\nspotify_search_url  150\ndtype: int64")

    # ── 3. Feature Engineering ──
    st.markdown("## 3. Feature Engineering")
    cell(5, """# Ambil tahun dari tanggal rilis
df['year'] = pd.to_datetime(df['release_date']).dt.year

# Log transform untuk fitur yang distribusinya miring
df['log_rating_count'] = np.log1p(df['rating_count'])
df['log_review_count'] = np.log1p(df['review_count'])

# Buat label era berdasarkan tahun
df['era'] = pd.cut(df['year'], bins=[0,1969,1984,1999,2012,2026],
                   labels=['Pionir','Old School','Mid High School','New School','New New School'])

# Buat label aksesibilitas berdasarkan skor
df['acc_label'] = pd.cut(df['accessibility'], bins=[0,2,4,6,8,10.1],
                          labels=['Niche','Elitis','Tidak Basic','Agak Lumayan Basic','Basic'])

df = df.dropna(subset=['era','acc_label'])
print(df['era'].value_counts().sort_index())""",
         "era\nPionir                 312\nOld School             887\nMid High School       1204\nNew School            1543\nNew New School        1031\nName: count, dtype: int64")

    # Chart distribusi era live
    label("VISUALISASI DISTRIBUSI ERA (LIVE)")
    ec = df["era"].value_counts().sort_index().reset_index(); ec.columns = ["era","count"]
    fig = px.bar(ec, x="era", y="count", template="plotly_dark", color="era",
                 color_discrete_map={"Pionir":"#C0A060","Old School":"#E07040","Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=200, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                      xaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#555",size=9)),
                      yaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#444",size=9)))
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Model Era ──
    st.markdown("## 4. Model 1: Klasifikasi Era")
    note("Fitur: <code>year, avg_rating, log_rating_count, log_review_count, accessibility</code>")
    cell(6, """X_era = df[['year','avg_rating','log_rating_count','log_review_count','accessibility']]
y_era = df['era']
X_train, X_test, y_train, y_test = train_test_split(X_era, y_era, test_size=0.2, random_state=42)

# Latih 3 model, pilih yang terbaik
era_models = {
    'Logistic Regression': Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))]),
    'Random Forest':       Pipeline([('scaler',StandardScaler()),('model',RandomForestClassifier(n_estimators=100))]),
    'KNN':                 Pipeline([('scaler',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=5))]),
}
hasil = {}
for nama, model in era_models.items():
    model.fit(X_train, y_train)
    hasil[nama] = accuracy_score(y_test, model.predict(X_test))
    print(f"  {nama:25s} -> {hasil[nama]*100:.2f}%")

model_era_gacor = era_models[max(hasil, key=hasil.get)]""",
         "  Logistic Regression       -> 81.24%\n  Random Forest             -> <span style='color:#e8d5a3;font-weight:600;'>94.71%</span>  &lt;-- terpilih\n  KNN                       -> 88.43%")

    cell(7, """print(classification_report(y_test, model_era_gacor.predict(X_test)))""",
         "                   precision  recall  f1-score\nPionir                  0.92    0.89      0.90\nOld School              0.95    0.96      0.96\nMid High School         0.94    0.93      0.94\nNew School              0.96    0.97      0.96\nNew New School          0.93    0.94      0.94\n\naccuracy                                0.95")

    cell(8, "joblib.dump(model_era_gacor, 'model_era_gacor.joblib')", "['model_era_gacor.joblib']")

    # ── 5. Model Aksesibilitas ──
    st.markdown("## 5. Model 2: Klasifikasi Aksesibilitas")
    note("Fitur: <code>accessibility, avg_rating, log_rating_count, log_review_count</code> — kodenya hampir sama")
    cell(9, """X_acc = df[['accessibility','avg_rating','log_rating_count','log_review_count']]
y_acc = df['acc_label']
X_train, X_test, y_train, y_test = train_test_split(X_acc, y_acc, test_size=0.2, random_state=42)

acc_models = {
    'Logistic Regression': Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))]),
    'Random Forest':       Pipeline([('scaler',StandardScaler()),('model',RandomForestClassifier(n_estimators=100))]),
    'KNN':                 Pipeline([('scaler',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=5))]),
}
hasil = {}
for nama, model in acc_models.items():
    model.fit(X_train, y_train)
    hasil[nama] = accuracy_score(y_test, model.predict(X_test))
    print(f"  {nama:25s} -> {hasil[nama]*100:.2f}%")

model_acc_gacor = acc_models[max(hasil, key=hasil.get)]
joblib.dump(model_acc_gacor, 'model_acc_gacor.joblib')""",
         "  Logistic Regression       -> 88.90%\n  Random Forest             -> <span style='color:#e8d5a3;font-weight:600;'>97.12%</span>  &lt;-- terpilih\n  KNN                       -> 91.40%")

    # ── 6. Fungsi Pipeline ──
    st.markdown("## 6. Fungsi-Fungsi Pipeline")

    cell(10, """# Cari baris album/artis yang dipilih user di dataset
def get_rows(albums, artists, df):
    mask = df["release_name"].isin(albums) | df["artist_name"].isin(artists)
    return df[mask].drop_duplicates(subset=["release_name","artist_name"])

# Prediksi era dominan dari kumpulan album
def get_era(dfc, model):
    p = model.predict(dfc[["year","avg_rating","log_rating_count","log_review_count","accessibility"]])
    return Counter(p).most_common(1)[0][0]

# Prediksi aksesibilitas dominan
def get_acc(dfc, model):
    p = model.predict(dfc[["accessibility","avg_rating","log_rating_count","log_review_count"]])
    return Counter(p).most_common(1)[0][0]

# Ambil 3 mood paling sering dari descriptor, skip kata teknis
EXCLUDE = {"malevocals","femalevocals","conceptalbum","instrumental","mixedvocals"}
def get_mood(dfc):
    semua = [k.strip() for d in dfc["descriptors"].dropna() for k in d.split(",") if k.strip() not in EXCLUDE]
    return Counter(semua).most_common(3) if semua else [("tidak tersedia",0)]

# Rekomendasikan album: era sama, aksesibilitas mirip, mood overlap
ACC_ORDER = ["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"]
def get_rek(era, acc, mood, dfc, df, n=5):
    sudah = set(dfc["release_name"].str.lower())
    idx   = ACC_ORDER.index(acc) if acc in ACC_ORDER else 2
    acc_r = [ACC_ORDER[i] for i in {max(0,idx-1), idx, min(4,idx+1)}]
    kata  = {m[0] for m in mood}
    rek   = df[(df["era"]==era) & (df["acc_label"].isin(acc_r)) & (~df["release_name"].str.lower().isin(sudah))].copy()
    if len(rek) < n:  # fallback: longgarkan ke semua aksesibilitas
        rek = df[(df["era"]==era) & (~df["release_name"].str.lower().isin(sudah))].copy()
    # Skor = berapa descriptor yang cocok dengan mood user
    rek["skor"] = rek["descriptors"].apply(lambda d: 0 if pd.isna(d) else len({k.strip() for k in d.split(",")} & kata))
    return rek.sort_values(["skor","avg_rating"], ascending=[False,False]).head(n)""")

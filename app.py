import html
import os
import nbformat
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
html, body, [class*="css"]  { font-family:'DM Sans',sans-serif; font-size:16px; }
[data-testid="stSidebar"]   { background:#0a0a0a !important; border-right:1px solid #1a1a1a !important; }
[data-testid="stAppViewContainer"] { background:#080808; }
[data-testid="stHeader"]    { background:transparent; }

/* Konten tab — beri napas di atas */
[data-testid="stTabsContent"] > div { padding-top:2rem !important; }

/* Kartu metrik */
[data-testid="stMetric"]    { background:#0f0f0f; border:1px solid #1e1e1e; border-radius:6px; padding:1.1rem 1.3rem; }
[data-testid="stMetricLabel"] { font-family:'DM Mono',monospace !important; font-size:0.7rem !important; letter-spacing:0.18em !important; color:#a8a8a8 !important; }
[data-testid="stMetricValue"] { font-family:'DM Serif Display',serif !important; color:#e8d5a3 !important; font-size:1.9rem !important; }

/* Tombol utama */
.stButton>button[kind="primary"] { background:#e8d5a3 !important; color:#080808 !important; font-family:'DM Mono',monospace !important; font-size:0.8rem !important; letter-spacing:0.1em !important; border:none !important; border-radius:3px !important; padding:0.6rem 1.4rem !important; }

/* Tag multiselect */
.stMultiSelect [data-baseweb="tag"] { background:#181818 !important; border:1px solid #2a2a2a !important; font-family:'DM Mono',monospace !important; color:#e8d5a3 !important; border-radius:3px !important; }

/* Tab navigasi */
.stTabs [data-baseweb="tab-list"] { border-bottom:1px solid #1a1a1a !important; gap:0.2rem; }
.stTabs [data-baseweb="tab"]       { font-family:'DM Mono',monospace !important; font-size:0.72rem !important; color:#8a8a8a !important; padding:0.6rem 1rem !important; letter-spacing:0.08em; }
.stTabs [aria-selected="true"]     { color:#e8d5a3 !important; border-bottom-color:#e8d5a3 !important; }

/* Input & slider */
[data-testid="stSlider"] { padding:0.3rem 0 0.8rem; }
p, li, .stMarkdown p { font-size:1rem !important; line-height:1.85; }
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
    "Pionir":          {"emoji":"🎷","rentang":"~1920–1969","warna":"#C0A060","desc":"Era perintisan musik modern. Para legenda membangun fondasi dari nol.","tokoh":"The Beatles · Bob Dylan · Miles Davis"},
    "Old School":      {"emoji":"🎸","rentang":"1970–1984", "warna":"#E07040","desc":"Era kejayaan rock, soul, dan funk. Ikonik dan tidak lekang waktu.","tokoh":"Led Zeppelin · Pink Floyd · Stevie Wonder"},
    "Mid High School": {"emoji":"📼","rentang":"1985–1999", "warna":"#5080D0","desc":"Era alternatif dan grunge. Penuh eksperimen dan keberanian.","tokoh":"Nirvana · Radiohead · Nas · Björk"},
    "New School":      {"emoji":"💿","rentang":"2000–2012", "warna":"#40B080","desc":"Era digital awal. Gerakan indie mulai bersinar luas.","tokoh":"My Chemical Romance · Kanye West · Sufjan Stevens"},
    "New New School":  {"emoji":"📱","rentang":"2013–kini", "warna":"#A050E0","desc":"Era streaming. Musik semakin beragam dan mudah dijangkau.","tokoh":"Kendrick Lamar · Frank Ocean · Laufey"},
}
ACC_INFO = {
    "Niche":             {"emoji":"🔬","warna":"#3050A0","singkat":"Sangat Niche","desc":"Selera Anda sangat spesifik dan jarang dikenal khalayak umum."},
    "Elitis":            {"emoji":"🎓","warna":"#6040B0","singkat":"Elitis","desc":"Anda cenderung menyukai musik underground bermutu tinggi."},
    "Tidak Basic":       {"emoji":"🎸","warna":"#208060","singkat":"Tidak Basic","desc":"Selera Anda berada di luar arus utama, namun masih terjangkau."},
    "Agak Lumayan Basic":{"emoji":"🎧","warna":"#808020","singkat":"Cukup Populer","desc":"Anda menyukai musik yang cukup dikenal tanpa terlalu mainstream."},
    "Basic":             {"emoji":"📻","warna":"#A04020","singkat":"Sangat Populer","desc":"Anda menyukai musik yang sangat populer dan dikenal luas."},
}
MOOD_DESC = {
    "melancholic":"Bernuansa sedih namun indah dan menyentuh.","atmospheric":"Membangun suasana dan ruang yang mendalam.",
    "introspective":"Mengajak merefleksikan diri ke dalam.","anxious":"Bernuansa gelisah dan penuh ketegangan.",
    "cold":"Terasa dingin dan berjarak, namun tetap menarik.","energetic":"Penuh semangat dan energi.",
    "dark":"Mengandung unsur kegelapan yang intens.","epic":"Berskala besar dan terasa megah.",
    "psychedelic":"Membuka persepsi dengan cara yang unik.","hypnotic":"Memiliki daya hipnotis yang kuat.",
    "lonely":"Terasa sunyi dan penuh kerinduan.","romantic":"Menghadirkan kehangatan cinta.",
    "mellow":"Santai dan nyaman untuk didengarkan.",
}
ACC_ORDER = ["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"]
EXCLUDE   = {"malevocals","femalevocals","conceptalbum","instrumental","mixedvocals"}


# === FUNGSI HELPER UI ===
def label(text):
    # Label kecil gaya DM Mono — dipakai sebagai sub-judul bagian
    st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#b0b0b0;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.5rem;'>{text}</p>", unsafe_allow_html=True)

def card(content, warna="#1e1e1e", border_left=None):
    # Kartu gelap dengan aksen warna opsional di sisi kiri
    bl = f"border-left:3px solid {border_left};" if border_left else ""
    st.markdown(f"<div style='background:#0f0f0f;border:1px solid {warna}33;{bl}border-radius:4px;padding:1.3rem;'>{content}</div>", unsafe_allow_html=True)

def bar_dark(data, x, y, h=280):
    # Bagan batang horizontal dengan tema gelap
    fig = px.bar(data, x=x, y=y, orientation="h", template="plotly_dark",
                 color=x, color_continuous_scale=["#1a1a1a","#e8d5a3"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, coloraxis_showscale=False, height=h,
                      margin=dict(l=0,r=0,t=6,b=0),
                      xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#aaa",size=10)),
                      yaxis=dict(tickfont=dict(color="#ddd",size=10), categoryorder="total ascending"))
    return fig

def divider(title):
    # Pemisah antar-segmen — garis tipis + judul bagian di bawahnya
    st.markdown(f"<div style='margin:2.5rem 0 1.3rem;padding-top:1.8rem;border-top:1px solid #161616;'><p style='font-family:DM Mono,monospace;font-size:0.68rem;color:#999;letter-spacing:0.22em;text-transform:uppercase;margin:0;'>{title}</p></div>", unsafe_allow_html=True)


# === FUNGSI ML ===
def get_rows(albums, artists):
    # Ambil baris dari dataset sesuai pilihan pengguna
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
    # Ambil 3 mood descriptor yang paling sering muncul, abaikan kata teknis
    semua = [k.strip() for d in dfc["descriptors"].dropna() for k in d.split(",") if k.strip() not in EXCLUDE]
    return Counter(semua).most_common(3) if semua else [("tidak tersedia",0)]

def get_rek(era, acc, mood, dfc, n=5):
    # Rekomendasikan album: era sama, aksesibilitas mirip, mood overlap
    sudah = set(dfc["release_name"].str.lower())
    idx   = ACC_ORDER.index(acc) if acc in ACC_ORDER else 2
    acc_r = [ACC_ORDER[i] for i in {max(0,idx-1), idx, min(4,idx+1)}]
    kata  = {m[0] for m in mood}
    rek   = df[(df["era"]==era) & (df["acc_label"].isin(acc_r)) & (~df["release_name"].str.lower().isin(sudah))].copy()
    if len(rek) < n:  # fallback: longgarkan filter aksesibilitas
        rek = df[(df["era"]==era) & (~df["release_name"].str.lower().isin(sudah))].copy()
    rek["skor"] = rek["descriptors"].apply(lambda d: 0 if pd.isna(d) else len({k.strip() for k in d.split(",")} & kata))
    return rek.sort_values(["skor","avg_rating"], ascending=[False,False]).head(n)


# === SIDEBAR ===
with st.sidebar:
    # Judul dan branding aplikasi
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:2.4rem;color:#f0ede8;letter-spacing:-0.03em;line-height:1;margin:1rem 0 0.2rem;\">REQ<sup style='font-size:0.3em;vertical-align:super;color:#e8d5a3;'>✦</sup></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#888;letter-spacing:0.16em;margin:0 0 1.2rem;'>MUSIC MACHINE LEARNING</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border:0;border-top:1px solid #1a1a1a;margin:0 0 1.2rem;'>", unsafe_allow_html=True)

    # Daftar navigasi halaman
    label("Penjelasan Singkat")
    for ikon, judul, ket in [
        ("✦","Analisis & Prediksi","Masukkan album atau artis favorit Anda."),
        ("◎","Pengantar","Gambaran umum cara kerja sistem."),
        ("▤","Dataset","Jelajahi dataset RateYourMusic."),
        ("◈","Tentang Saya","Profil pengembang aplikasi."),
        ("⌥","Kode Proyek","Notebook pelatihan model secara lengkap."),
    ]:
        st.markdown(f"<div style='padding:0.6rem 0.8rem;margin-bottom:0.3rem;border-radius:3px;border:1px solid #161616;background:#0c0c0c;'><p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#e8d5a3;margin:0 0 0.15rem;'>{ikon} {judul}</p><p style='font-size:0.82rem;color:#aaa;margin:0;'>{ket}</p></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:0;border-top:1px solid #1a1a1a;margin:1rem 0;'>", unsafe_allow_html=True)

    # Statistik ringkas dataset
    label("Statistik Dataset")
    for nama, nilai in [("Total Album",f"{len(df):,}"),("Total Artis",f"{df['artist_name'].nunique():,}"),
                        ("Rentang Tahun",f"{int(df['year'].min())}–{int(df['year'].max())}"),("Rata-rata Rating",f"{df['avg_rating'].mean():.2f}")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #111;'><span style='font-size:0.82rem;color:#aaa;'>{nama}</span><span style='font-family:DM Mono,monospace;font-size:0.82rem;color:#e8d5a3;'>{nilai}</span></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Informasi akurasi model
    label("Akurasi Model")
    for nm, val in [("Model Era — Random Forest","94.71 %"),("Model Aksesibilitas — Random Forest","97.12 %")]:
        st.markdown(f"<div style='padding:0.5rem 0.7rem;margin-bottom:0.3rem;border-radius:3px;border:1px solid #161616;background:#0c0c0c;'><p style='font-family:DM Mono,monospace;font-size:0.6rem;color:#999;margin:0 0 0.1rem;'>{nm}</p><p style='font-family:DM Mono,monospace;font-size:0.75rem;color:#e8d5a3;margin:0;'>{val}</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.55rem;color:#666;letter-spacing:0.12em;line-height:2;border-top:1px solid #1a1a1a;padding-top:1rem;'>DATA · RateYourMusic<br>MODEL · scikit-learn<br>UI · Streamlit</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB UTAMA — Analisis & Prediksi ditampilkan paling depan
# ══════════════════════════════════════════════════════
Analisis, Introduction, Dataset, Kode_Proyek, About_Me = st.tabs([
    "✦  Analisis & Prediksi", "◎  Pengantar", "▤  Dataset", "⌥  Kode Proyek", "◈  Tentang Saya"
])


# ══════════════════════════
# HALAMAN 1 — ANALISIS & PREDIKSI
# ══════════════════════════
with Analisis:
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Analisis &amp; Prediksi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#999;letter-spacing:0.15em;margin-bottom:2rem;'>PILIH ALBUM ATAU ARTIS — SISTEM AKAN MENGANALISIS SELERA MUSIK ANDA</p>", unsafe_allow_html=True)

    # Ambil semua nama album dan artis unik dari dataset
    ALL_ALBUMS  = sorted(df["release_name"].dropna().unique(), key=str.lower)
    ALL_ARTISTS = sorted(df["artist_name"].dropna().unique(), key=str.lower)

    # Input pilihan pengguna
    c1,c2 = st.columns(2)
    sel_albums  = c1.multiselect("🎵 Album yang Disukai",  ALL_ALBUMS,  placeholder="Ketik atau pilih...")
    sel_artists = c2.multiselect("🎤 Artis yang Disukai",  ALL_ARTISTS, placeholder="Ketik atau pilih...")
    n_rek = st.slider("Jumlah Rekomendasi", 3, 10, 5)

    # Tombol analisis
    _, bc, _ = st.columns([2,1,2])
    run = bc.button("✦  Analisis Sekarang", type="primary", use_container_width=True)

    if run:
        if not sel_albums and not sel_artists:
            st.warning("Silakan pilih minimal satu album atau artis terlebih dahulu.")
        else:
            # Cari data album yang sesuai di dataset
            dfc = get_rows(sel_albums, sel_artists)
            if dfc.empty:
                st.error("Data tidak ditemukan. Pastikan nama album atau artis sesuai dengan dataset.")
            else:
                # Jalankan prediksi dan rekomendasi
                era_fav  = get_era(dfc)
                acc_fav  = get_acc(dfc)
                top_mood = get_mood(dfc)
                df_rek   = get_rek(era_fav, acc_fav, top_mood, dfc, n=n_rek)
                ei, ai   = ERA_INFO.get(era_fav,{}), ACC_INFO.get(acc_fav,{})

                st.markdown(f"<p style='font-size:0.9rem;color:#aaa;font-style:italic;margin-top:1rem;'>Analisis didasarkan pada {len(dfc)} album yang ditemukan dalam dataset.</p>", unsafe_allow_html=True)

                # ── PREDIKSI ERA & AKSESIBILITAS ──────────────────────
                # Dua model ML memprediksi era dan tingkat aksesibilitas
                # selera musik berdasarkan fitur album yang dipilih.
                divider("Hasil Prediksi — Era dan Aksesibilitas Selera Musik Anda")

                ce,ca = st.columns(2)
                # Kartu hasil prediksi era musik
                with ce:
                    card(f"""
                    <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ei['warna']};letter-spacing:0.2em;margin:0 0 0.6rem;'>MODEL 1 — ERA MUSIK</p>
                    <div style='font-size:1.8rem;'>{ei['emoji']}</div>
                    <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f0ede8;margin:0;">{era_fav}</h2>
                    <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ei['warna']};margin:0.3rem 0 0.8rem;'>{ei['rentang']}</p>
                    <p style='font-size:0.88rem;color:#c8c8c8;line-height:1.7;margin:0 0 0.4rem;'>{ei['desc']}</p>
                    <p style='font-size:0.8rem;color:#999;margin:0;'>{ei['tokoh']}</p>
                    """, warna=ei['warna'], border_left=ei['warna'])
                # Kartu hasil prediksi aksesibilitas dengan skala visual
                with ca:
                    skala = "".join(f"<div style='flex:1;height:5px;background:{ai['warna'] if lv==acc_fav else '#1e1e1e'};border-radius:2px;'></div>" for lv in ACC_ORDER)
                    card(f"""
                    <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ai['warna']};letter-spacing:0.2em;margin:0 0 0.6rem;'>MODEL 2 — AKSESIBILITAS SELERA</p>
                    <div style='font-size:1.8rem;'>{ai['emoji']}</div>
                    <h2 style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#f0ede8;margin:0;">{acc_fav}</h2>
                    <p style='font-family:DM Mono,monospace;font-size:0.65rem;color:{ai['warna']};margin:0.3rem 0 0.8rem;'>{ai['singkat']}</p>
                    <p style='font-size:0.88rem;color:#c8c8c8;line-height:1.7;margin:0 0 0.8rem;'>{ai['desc']}</p>
                    <p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#999;margin:0 0 0.4rem;'>NICHE &lt;——&gt; BASIC</p>
                    <div style='display:flex;gap:0.3rem;'>{skala}</div>
                    """, warna=ai['warna'], border_left=ai['warna'])

                # ── MOOD DOMINAN ───────────────────────────────────────
                # Tiga mood paling sering muncul dari deskriptor album
                # mencerminkan nuansa emosional dominan selera musik Anda.
                divider("Karakter Mood Musik Anda — Diambil dari Deskriptor Album")
                # Kolom dibuat sekali di luar loop agar ketiga kartu sejajar
                mood_cols = st.columns(3)
                for i,(mood,count) in enumerate(top_mood):
                    mood_cols[i].markdown(f"<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.2rem;height:100%;'><p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#e8d5a3;margin:0 0 0.4rem;'>#{i+1}</p><h3 style=\"font-family:'DM Serif Display',serif;font-size:1.3rem;color:#f0ede8;margin:0 0 0.3rem;text-transform:capitalize;\">{mood}</h3><p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#999;margin:0 0 0.6rem;'>muncul {count}x</p><p style='font-size:0.88rem;color:#bbb;line-height:1.6;margin:0;'>{MOOD_DESC.get(mood,'Deskriptor unik yang jarang ditemukan.')}</p></div>", unsafe_allow_html=True)

                # ── REKOMENDASI ALBUM ──────────────────────────────────
                # Album direkomendasikan berdasarkan kesamaan era, aksesibilitas,
                # dan kemiripan mood. Diurutkan dari skor kesesuaian tertinggi.
                divider(f"Rekomendasi Album — {n_rek} Album Terpilih untuk Anda")
                for i,(_,r) in enumerate(df_rek.iterrows()):
                    vibes = " · ".join([k.strip() for k in r.get("descriptors","").split(",") if k.strip() not in EXCLUDE][:4]) if pd.notna(r.get("descriptors")) else ""
                    url   = r.get("spotify_search_url","")
                    spot  = f'<a href="{url}" target="_blank" style="display:inline-block;background:#1DB954;color:#000;font-family:DM Mono,monospace;font-size:0.65rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:20px;text-decoration:none;margin-top:0.6rem;">&#9654; Buka di Spotify</a>' if pd.notna(url) and str(url).startswith("http") else ""
                    st.markdown(f"""<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1rem 1.3rem;margin-bottom:0.5rem;'>
                        <span style='font-family:DM Mono,monospace;font-size:0.6rem;color:#777;'>#{i+1:02d}</span>
                        <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#f0ede8;margin-left:0.5rem;">{r['release_name']}</span>
                        <span style='font-size:0.85rem;color:#aaa;margin-left:0.5rem;'>{r['artist_name']} &middot; {int(r['year'])}</span>
                        <div style='display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.82rem;color:#bbb;'>
                            <span>&#9733; <b style='color:#e8d5a3;'>{r['avg_rating']:.2f}</b></span>
                            <span>{r.get('primary_genres','—')}</span>
                            <span style='color:#999;'>{r['acc_label']}</span>
                        </div>
                        {f"<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#888;margin:0.4rem 0 0;'>{vibes}</p>" if vibes else ''}
                        {spot}
                    </div>""", unsafe_allow_html=True)

                # ── DATA ALBUM PENGGUNA ────────────────────────────────
                # Daftar album dan artis yang berhasil ditemukan dalam dataset
                # berdasarkan pilihan pengguna, beserta data lengkapnya.
                divider(f"Data Album Anda — {len(dfc)} Album Ditemukan dalam Dataset")
                st.dataframe(dfc[["release_name","artist_name","year","avg_rating","accessibility","era","acc_label","primary_genres"]],
                    use_container_width=True,
                    column_config={
                        "avg_rating":   st.column_config.NumberColumn("Rating", format="%.2f"),
                        "accessibility":st.column_config.ProgressColumn("Aksesibilitas", min_value=0, max_value=10),
                    })


# ══════════════════════════
# HALAMAN 2 — PENGANTAR
# ══════════════════════════
with Introduction:
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:5rem;color:#f0ede8;letter-spacing:-0.03em;line-height:1;margin:0 0 0.5rem;\">REQ<sup style='font-size:0.28em;vertical-align:super;color:#e8d5a3;'>✦</sup></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#999;letter-spacing:0.2em;margin:0 0 2rem;'>SISTEM REKOMENDASI MUSIK BERBASIS MACHINE LEARNING</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1rem;color:#bbb;max-width:560px;line-height:1.9;margin-bottom:2rem;'>Masukkan album atau artis favorit Anda — REQ akan memprediksi era musik, tingkat aksesibilitas selera, dan mood dominan Anda, lalu memberikan rekomendasi album yang paling sesuai.</p>", unsafe_allow_html=True)

    # Metrik ringkas dataset
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Album",      f"{len(df):,}")
    c2.metric("Rentang Tahun",    f"{int(df['year'].min())}–{int(df['year'].max())}")
    c3.metric("Rata-rata Rating", f"{df['avg_rating'].mean():.2f}")
    c4.metric("Total Artis",      f"{df['artist_name'].nunique():,}")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Visualisasi distribusi era dalam dataset
    label("DISTRIBUSI ERA DALAM DATASET")
    ec = df["era"].value_counts().sort_index().reset_index(); ec.columns = ["era","count"]
    fig = px.bar(ec, x="era", y="count", template="plotly_dark", color="era",
                 color_discrete_map={"Pionir":"#C0A060","Old School":"#E07040","Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=200, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                      xaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#aaa",size=10)),
                      yaxis=dict(gridcolor="#1a1a1a",tickfont=dict(color="#999",size=9)))
    st.plotly_chart(fig, use_container_width=True)

    # Tiga langkah cara kerja sistem
    label("CARA KERJA")
    for col, num, judul, isi in zip(st.columns(3),
        ["01","02","03"],
        ["Input","Prediksi ML","Rekomendasi"],
        ["Pilih album atau artis yang tersedia dalam dataset.",
         "Dua model memprediksi era dan aksesibilitas selera musik Anda.",
         "Album direkomendasikan berdasarkan kecocokan era dan mood."]):
        col.markdown(f"<div style='background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.3rem;'><p style='font-family:DM Mono,monospace;font-size:0.55rem;color:#777;margin:0 0 0.4rem;'>{num}</p><p style=\"font-family:'DM Serif Display',serif;font-size:1rem;color:#e8d5a3;margin:0 0 0.4rem;\">{judul}</p><p style='font-size:0.88rem;color:#bbb;line-height:1.6;margin:0;'>{isi}</p></div>", unsafe_allow_html=True)


# ══════════════════════════
# HALAMAN 3 — DATASET
# ══════════════════════════
with Dataset:
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Dataset</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#999;letter-spacing:0.15em;margin-bottom:2rem;'>SUMBER · RATEYOURMUSIC — 5.000+ ALBUM</p>", unsafe_allow_html=True)

    # Metrik ringkas dataset
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Album",            f"{len(df):,}")
    m2.metric("Tahun",                  f"{int(df['year'].min())}–{int(df['year'].max())}")
    m3.metric("Rata-rata Rating",       f"{df['avg_rating'].mean():.2f}")
    m4.metric("Rata-rata Aksesibilitas",f"{df['accessibility'].mean():.1f}/10")

    st.markdown("<hr style='border:0;border-top:1px solid #1a1a1a;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)

    # ── Tabel data — ditampilkan lebih dahulu karena lebih sering dibutuhkan
    label("Jelajahi Data — Filter dan Telusuri Album")
    f1,f2,f3 = st.columns([2,2,1])
    era_f = f1.multiselect("Filter Era", df["era"].cat.categories.tolist())
    acc_f = f2.multiselect("Filter Aksesibilitas", ACC_ORDER)
    n_r   = f3.number_input("Baris", 10, 500, 50, 10)

    # Terapkan filter ke salinan dataset
    df_show = df.copy()
    if era_f: df_show = df_show[df_show["era"].isin(era_f)]
    if acc_f: df_show = df_show[df_show["acc_label"].isin(acc_f)]

    label(f"{len(df_show):,} baris ditemukan")
    st.dataframe(df_show[["release_name","artist_name","year","avg_rating","rating_count","accessibility","era","acc_label","primary_genres"]].head(n_r),
        use_container_width=True, height=420,
        column_config={
            "release_name": st.column_config.TextColumn("Album"),
            "artist_name":  st.column_config.TextColumn("Artis"),
            "year":         st.column_config.NumberColumn("Tahun", format="%d"),
            "avg_rating":   st.column_config.NumberColumn("Rating", format="%.2f"),
            "accessibility":st.column_config.ProgressColumn("Aksesibilitas", min_value=0, max_value=10),
        })

    st.markdown("<hr style='border:0;border-top:1px solid #1a1a1a;margin:1.5rem 0 1rem;'>", unsafe_allow_html=True)

    # ── Visualisasi distribusi dan perbandingan data
    label("Visualisasi Dataset")
    cl, cr = st.columns(2)
    with cl:
        label("TOP 10 ARTIS — TOTAL RATING")
        top_a = df.groupby("artist_name")["rating_count"].sum().nlargest(10).reset_index()
        st.plotly_chart(bar_dark(top_a,"rating_count","artist_name"), use_container_width=True)

        label("JUMLAH RATING vs RATA-RATA RATING")
        fig2 = px.scatter(df, x="log_rating_count", y="avg_rating", color="era", opacity=0.45,
                          template="plotly_dark", hover_data=["release_name","artist_name"],
                          color_discrete_map={"Pionir":"#C0A060","Old School":"#E07040","Mid High School":"#5080D0","New School":"#40B080","New New School":"#A050E0"})
        fig2.update_traces(marker_size=3)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=260, margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                           xaxis=dict(gridcolor="#1e1e1e",tickfont=dict(color="#aaa",size=9)),
                           yaxis=dict(gridcolor="#1e1e1e",tickfont=dict(color="#aaa",size=9)))
        st.plotly_chart(fig2, use_container_width=True)

    with cr:
        label("TOP GENRE")
        semua_genre = [x.strip() for g in df["primary_genres"].dropna() for x in g.split(",")]
        gdf = pd.DataFrame(Counter(semua_genre).most_common(12), columns=["genre","count"])
        st.plotly_chart(bar_dark(gdf,"count","genre"), use_container_width=True)

        label("TOP 10 ALBUM — RATA-RATA RATING")
        top10 = df.nlargest(10,"avg_rating")[["release_name","avg_rating"]].copy()
        top10["release_name"] = top10["release_name"].str[:24]
        st.plotly_chart(bar_dark(top10,"avg_rating","release_name"), use_container_width=True)


# ══════════════════════════
# HALAMAN 4 — TENTANG SAYA
# ══════════════════════════
with About_Me:
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:2rem;\">Tentang Saya</h1>", unsafe_allow_html=True)
    cf, ci = st.columns([1,2], gap="large")
    with cf:
        st.markdown("<div style='border:1px solid #1e1e1e;border-radius:4px;overflow:hidden;'><img src='https://raw.githubusercontent.com/MF-KIMPOEL/projek_ML_Musik/main/hafidz.jpeg' style='width:100%;display:block;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        label("TEKNOLOGI YANG DIGUNAKAN")
        badges = " ".join(f"<span style='font-family:DM Mono,monospace;font-size:0.65rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;display:inline-block;margin:0.2rem 0.1rem;'>{t}</span>"
                          for t in ["Python","Streamlit","scikit-learn","pandas","Plotly","joblib","RateYourMusic"])
        st.markdown(f"<div style='line-height:2;'>{badges}</div>", unsafe_allow_html=True)
    with ci:
        st.markdown("<h2 style=\"font-family:'DM Serif Display',serif;font-size:2rem;color:#f0ede8;margin:0 0 0.3rem;\">RIGEL AMADEUS VOLKER</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.65rem;color:#999;letter-spacing:0.2em;margin:0 0 2rem;'>SISWA &middot; RPL &middot; SMKN PURBALINGGA</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.95rem;color:#bbb;line-height:1.9;margin-bottom:1.5rem;'>Nama saya Rigel Amadeus Volker, siswa SMKN Purbalingga jurusan RPL kelas 11. Saya menyukai musik dan film — oleh karena itu saya mengembangkan proyek machine learning ini sebagai bentuk eksplorasi minat tersebut.</p>", unsafe_allow_html=True)
        label("KONTAK")
        st.markdown("<p style='font-size:0.9rem;color:#aaa;line-height:2.4;'>&#128231; rigel123@gmail.com<br>&#128025; github.com/rigelgithub<br>&#127925; last.fm/gapunya/gapunya123</p>", unsafe_allow_html=True)
        st.markdown("<hr style='border:0;border-top:1px solid #1a1a1a;margin:1.2rem 0;'>", unsafe_allow_html=True)
        label("TENTANG PROYEK")
        st.markdown("<p style='font-size:0.95rem;color:#bbb;line-height:1.9;'>REQ adalah sistem rekomendasi album berbasis Machine Learning menggunakan dataset dari RateYourMusic. Model mengklasifikasikan era dan aksesibilitas selera musik pengguna, lalu merekomendasikan album berdasarkan mood dan era yang paling sesuai.</p>", unsafe_allow_html=True)
# ══════════════════════════
# HALAMAN 5 — KODE PROYEK
# Membaca proyek_reyal.ipynb langsung via nbformat sehingga
# output nyata tiap sel (gambar, tabel, teks) ikut ditampilkan.
# ══════════════════════════
with Kode_Proyek:
    st.markdown("<h1 style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;margin-bottom:0.2rem;\">Kode Proyek</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:#999;letter-spacing:0.15em;margin-bottom:2rem;'>proyek reyal.ipynb — Kode Model ML</p>", unsafe_allow_html=True)

    nb_path = os.path.join(os.path.dirname(__file__), "proyek reyal.ipynb")

    if not os.path.exists(nb_path):
        st.warning("File proyek_reyal.ipynb tidak ditemukan. Pastikan file berada satu direktori dengan app.py.")
    else:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for i, cell in enumerate(nb.cells):
            tipe      = cell.cell_type
            label_sel = "🟦 Markdown" if tipe == "markdown" else "🟩 Code"

            # Header tiap sel: nomor urut dan tipe sel
            st.markdown(
                f"""<div style="background:#1e2e22;border-left:3px solid #2fa05a;
                padding:6px 14px;border-radius:6px;margin-bottom:4px;">
                <span style="color:#7ab8f5;font-size:0.75rem;font-weight:600;">Sel [{i+1}]</span>
                <span style="color:rgba(255,255,255,0.7);font-size:0.75rem;">&nbsp;·&nbsp;{label_sel}</span>
                </div>""",
                unsafe_allow_html=True
            )

            if tipe == "markdown":
                # Sel teks/narasi — render sebagai Markdown biasa
                st.markdown(cell.source)

            elif tipe == "code":
                # Sel kode — tampilkan kode dengan syntax highlighting
                st.code(cell.source, language="python")

                # Iterasi seluruh output yang dihasilkan sel tersebut
                for output in cell.get("outputs", []):

                    # Output teks dari print() atau stderr
                    if output.output_type == "stream":
                        teks = html.escape(output.text)
                        st.markdown(f"""<div style="background:#0f1715;border-radius:6px;padding:10px 14px;font-family:monospace;font-size:0.82rem;color:#c5ecd1;white-space:pre-wrap;margin-top:4px;">{teks}</div>""", unsafe_allow_html=True)

                    elif output.output_type in ("display_data", "execute_result"):
                        # Output gambar dari matplotlib/plotly
                        if "image/png" in output.data:
                            import base64
                            img_data = output.data["image/png"]
                            st.markdown(f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;border-radius:8px;margin-top:6px;">', unsafe_allow_html=True)
                        # Output tabel dari pandas (df.head(), df.describe(), dsb.)
                        elif "text/html" in output.data:
                            st.markdown(output.data["text/html"], unsafe_allow_html=True)
                        # Output teks biasa (print, repr)
                        elif "text/plain" in output.data:
                            teks = html.escape(output.data["text/plain"])
                            st.markdown(f"""<div style="background:#0f1715;border-radius:6px;padding:10px 14px;font-family:monospace;font-size:0.82rem;color:#c5ecd1;white-space:pre-wrap;margin-top:4px;">{teks}</div>""", unsafe_allow_html=True)

            st.markdown("<hr style='border:0.5px solid #1e2e22;margin:12px 0'>", unsafe_allow_html=True)

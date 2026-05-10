import numpy as np
import pandas as pd
import streamlit as st
import joblib
#untuk visualisasi
import plotly.express as px
#untuk menghitung frekuensi data
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Load model dan dataset
model_era   = joblib.load("model_era_gacor.joblib")
model_akses = joblib.load("model_acc_gacor.joblib")
df = pd.read_csv("rym.csv")

# Bikin kolom tambahan yang sama persis kaya waktu training model
# tujuannya supaya modelnya tidak bingung kalau inputnya beda
#dibenerin skalanya supaya lebih waras untuk dibaca model
df["year"]             = pd.to_datetime(df["release_date"]).dt.year
df["log_rating_count"] = np.log1p(df["rating_count"])
df["log_review_count"] = np.log1p(df["review_count"])
df["era"] = pd.cut(df["year"], bins=[0,1969,1984,1999,2012,2026],
                   labels=["Pionir","Old School","Mid High School","New School","New New School"])
df["acc_label"] = pd.cut(df["accessibility"], bins=[0,2,4,6,8,10.1],
                         labels=["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"])
df = df.dropna(subset=["era","acc_label"])

# Daftar pilihan buat dropdown
ALL_ALBUMS  = sorted(df["release_name"].dropna().unique().tolist(), key=str.lower)
ALL_ARTISTS = sorted(df["artist_name"].dropna().unique().tolist(), key=str.lower)

# Info tiap era buat ditampilin di kartu hasil
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
# Penjelasan singkat untuk beberapa mood
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
ACC_ORDER = ["Niche","Elitis","Tidak Basic","Agak Lumayan Basic","Basic"]
#dikecualikan karena bukan mood
EXCLUDE   = {"malevocals","femalevocals","conceptalbum","instrumental","mixedvocals"}


# Fungsi-fungsi inti
def get_rows(albums, artists):
    # Ambil baris dari dataset, album dan artis yang dipilih
    mask = df["release_name"].isin(albums) | df["artist_name"].isin(artists)
    return df[mask].drop_duplicates(subset=["release_name","artist_name"])

def get_era(df_c):
    # Prediksi era dominan dari album-album yang dipilih
    p = model_era.predict(df_c[["year","avg_rating","log_rating_count","log_review_count","accessibility"]])
    return Counter(p).most_common(1)[0][0]

def get_acc(df_c):
    # Prediksi tingkat aksesibilitas dominan
    p = model_akses.predict(df_c[["accessibility","avg_rating","log_rating_count","log_review_count"]])
    return Counter(p).most_common(1)[0][0]

def get_mood(df_c):
    # Hitung 3 mood/descriptor yang paling sering muncul
    semua = [k.strip() for d in df_c["descriptors"].dropna()
             for k in d.split(",") if k.strip() not in EXCLUDE]
    #beri 3 data ang palign sering muncul jika tidak ada keluar teks tidak tersedia
    return Counter(semua).most_common(3) if semua else [("tidak tersedia",0)]

def get_rek(era, acc, mood, df_c, n=5):
    # Kasih rekomendasi: era yang sama + aksesibilitas mirip + mood mirip, album yang sudah disukai diexclude
    sudah = set(df_c["release_name"].str.lower())
    #aksesibilitas yang sama atau paling tidak mirip
    idx   = ACC_ORDER.index(acc) if acc in ACC_ORDER else 2
    acc_r = [ACC_ORDER[i] for i in {max(0,idx-1),idx,min(4,idx+1)}]
    kata  = {m[0] for m in mood}
    #rekomendasi
    rek   = df[(df["era"]==era) & (df["acc_label"].isin(acc_r)) &
               (~df["release_name"].str.lower().isin(sudah))].copy()
    if len(rek) < n:
        rek = df[(df["era"]==era) & (~df["release_name"].str.lower().isin(sudah))].copy()
    # Skor = berapa banyak descriptor yang nyamain mood user
    rek["skor"] = rek["descriptors"].apply(
        lambda d: 0 if pd.isna(d) else len({k.strip() for k in d.split(",")} & kata))
    return rek.sort_values(["skor","avg_rating"], ascending=[False,False]).head(n)


# Page setup, mengatur tampilan dasar web
st.set_page_config(page_title="Projek ML Musik", page_icon="🎵", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "intro"

# Navbar atas (cuma dekor biar keren)
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:1.2rem 0;border-bottom:1px solid #1a1a1a;margin-bottom:2rem;">
    <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#e8d5a3;">REQ<sup style="font-size:0.5em;">✦</sup></div>
    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#2a2a2a;letter-spacing:0.25em;">
        INTRODUCTION &nbsp;||&nbsp; DATASET &nbsp;||&nbsp; RECOMMENDATION
    </div>
    <div style="min-width:80px;"></div>
</div>
""", unsafe_allow_html=True)

# Tombol navigasi
n1, n2, n3, n4, n5 = st.columns(5)
if n1.button("Introduction", use_container_width=True): st.session_state.page = "intro"
if n2.button("Dataset",      use_container_width=True): st.session_state.page = "dataset"
if n3.button("Analisis",     use_container_width=True): st.session_state.page = "analisis"
if n4.button("About Me",     use_container_width=True): st.session_state.page = "about"
if n5.button("Kode Proyek",  use_container_width=True): st.session_state.page = "kode"
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)



# HALAMAN 1: INTRO
#introduction
if st.session_state.page == "intro":
    st.markdown("""
    <div style="text-align:center;padding:4rem 0 2rem;">
        <h1 style="font-family:'DM Serif Display',serif;font-size:7rem;color:#f0ede8;letter-spacing:-0.04em;line-height:1;margin:0;">
            REQ<sup style="font-size:0.3em;vertical-align:super;">✦</sup>
        </h1>
        <p style="font-size:0.92rem;color:#777;max-width:480px;margin:1.5rem auto 0;line-height:1.9;">
            REQ adalah projek ML musik yang dapat memberi rekomendasi lagu, memprediksi mood, era dan berbagai hal lainnya.
                akurat dan dapat dipercaya tentunya.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3 highlight
    for label in ["4500+ ALBUM", "COMPLETE TAG", "TRUSTED"]:
        st.markdown(
            f"<div style='text-align:center;padding:1.8rem 0;border-top:1px solid #1a1a1a;'>"
            f"<span style=\"font-family:'DM Serif Display',serif;font-size:3rem;color:#f0ede8;\">{label}</span></div>",
            unsafe_allow_html=True,
        )


# HALAMAN 2: DATASET
elif st.session_state.page == "dataset":
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:4rem;color:#f0ede8;margin-bottom:0.3rem;">DATASET</h1>
    <p style="font-size:0.85rem;color:#666;margin-bottom:2rem;">
        Diambil dari RateYourMusic — 5000 album, terima kasih komunitas RYM.
    </p>
    """, unsafe_allow_html=True)

    # Metric ringkas: total album, rentang tahun, rata-rata rating & aksesibilitas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Album",       f"{len(df):,}")
    m2.metric("Tahun",             f"{int(df['year'].min())}–{int(df['year'].max())}")
    m3.metric("Avg Rating",        f"{df['avg_rating'].mean():.2f}")
    m4.metric("Avg Aksesibilitas", f"{df['accessibility'].mean():.1f}/10")

    #deep dive
    st.markdown("<h2 style='font-family:DM Serif Display,serif;font-size:2.5rem;color:#f0ede8;margin:2.5rem 0 1.5rem;'>DEEP DIVE</h2>", unsafe_allow_html=True)

    # visualisasi data
  #template chart
    def bar_dark(data, x, y, h=300):
        f = px.bar(data, x=x, y=y, orientation="h", template="plotly_dark",
                   color=x, color_continuous_scale=["#1a1a1a", "#e8d5a3"])
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False, coloraxis_showscale=False, height=h,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9)),
                        yaxis=dict(tickfont=dict(color="#aaa", size=10), categoryorder="total ascending"))
        return f

    # judul chart
    def cap(t, mt=0):
        st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#666;letter-spacing:0.15em;margin-top:{mt}rem;'>{t}</p>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

  #total rating
    with col_l:
        cap("TOP 10 ARTIS — TOTAL RATING")
        top_a = df.groupby("artist_name")["rating_count"].sum().nlargest(10).reset_index()
        st.plotly_chart(bar_dark(top_a, "rating_count", "artist_name"), use_container_width=True)

        # Scatter
        cap("RATING COUNT VS AVG RATING", mt=1)
        f2 = px.scatter(df, x="log_rating_count", y="avg_rating", color="era", opacity=0.4,
                        template="plotly_dark", hover_data=["release_name", "artist_name"])
        f2.update_traces(marker_size=3)
        f2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                         xaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9)),
                         yaxis=dict(gridcolor="#1e1e1e", tickfont=dict(color="#555", size=9)))
        st.plotly_chart(f2, use_container_width=True)

    with col_r:
        # Top 12 genre paling sering muncul
        semua_genre = [x.strip() for g in df["primary_genres"].dropna() for x in g.split(",")]
        gdf = pd.DataFrame(Counter(semua_genre).most_common(12), columns=["genre", "count"])
        cap("TOP GENRES")
        st.plotly_chart(bar_dark(gdf, "count", "genre"), use_container_width=True)

        # Top 10 album
        cap("TOP 10 ALBUMS BY AVG RATING", mt=1)
        top10 = df.nlargest(10, "avg_rating")[["release_name", "avg_rating"]]
        top10["release_name"] = top10["release_name"].str[:22]
        st.plotly_chart(bar_dark(top10, "avg_rating", "release_name"), use_container_width=True)


# HALAMAN 3: ANALISIS
elif st.session_state.page == "analisis":

    # Form input
    col1, col2 = st.columns(2)
    with col1:
        #multiselect
        sel_albums = st.multiselect(
            "🎵 Album yang kamu suka", options=ALL_ALBUMS,
            placeholder="Ketik atau pilih album...",
            help="searching man",
        )
    with col2:
        sel_artists = st.multiselect(
            "🎤 Artis favorit kamu", options=ALL_ARTISTS,
            placeholder="Ketik atau pilih artis...",
            help="searching maning man",
        )

    # Slider jumlah rekomendasi
    n_rek = st.slider("Jumlah rekomendasi", 3, 10, 5)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    _, bc, _ = st.columns([1, 2, 1])
    run = bc.button("Generate Recommendations", type="primary", use_container_width=True)

    # Output
    if run:
        # Validasi: minimal pilih satu
        if not sel_albums and not sel_artists:
            st.warning("Pilih minimal satu album atau artis dulu ya.")
        else:
            df_c = get_rows(sel_albums, sel_artists)
            if df_c.empty:
                st.error("Yah, gaketemu di dataset.")
            else:
                # Jalanin model + ambil mood + rekomendasi
                era_fav  = get_era(df_c)
                acc_fav  = get_acc(df_c)
                top_mood = get_mood(df_c)
                df_rek   = get_rek(era_fav, acc_fav, top_mood, df_c, n=n_rek)
                ei, ai = ERA_INFO.get(era_fav, {}), ACC_INFO.get(acc_fav, {})

                st.markdown(
                    f"<p style='font-size:0.85rem;color:#777;margin:1.5rem 0;font-style:italic;'>"
                    f"Berdasarkan {len(df_c)} album yang kepilih — {ai.get('singkat','')}.</p>",
                    unsafe_allow_html=True,
                )

                # era kanan aksesibilitas kiri
                ce, ca = st.columns(2)
                with ce:
                    st.markdown(f"""
                    <div style="border:1px solid {ei.get('warna','#333')}33;background:#0f0f0f;border-radius:4px;padding:1.5rem;min-height:220px;">
                        <p style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{ei.get('warna','#888')};letter-spacing:0.2em;margin:0 0 0.5rem;">MODEL 1 — ERA</p>
                        <div style="font-size:1.6rem;">{ei.get('emoji','')}</div>
                        <h2 style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#f0ede8;margin:0;">{era_fav}</h2>
                        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{ei.get('warna','#888')};margin:0 0 0.8rem;">{ei.get('rentang','')}</p>
                        <p style="font-size:0.83rem;color:#777;line-height:1.7;">{ei.get('desc','')}</p>
                        <p style="font-size:0.75rem;color:#555;margin-top:0.5rem;">{ei.get('tokoh','')}</p>
                    </div>""", unsafe_allow_html=True)

                with ca:
                    # Skala 5 kotak, posisi aksesibilitas user
                    skala = "".join(
                        f"<div style='flex:1;height:5px;background:{ai.get('warna','#444') if lv==acc_fav else '#1e1e1e'};border-radius:2px;'></div>"
                        for lv in ACC_ORDER
                    )
                    st.markdown(f"""
                    <div style="border:1px solid {ai.get('warna','#333')}33;background:#0f0f0f;border-radius:4px;padding:1.5rem;min-height:220px;">
                        <p style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{ai.get('warna','#888')};letter-spacing:0.2em;margin:0 0 0.5rem;">MODEL 2 — AKSESIBILITAS</p>
                        <div style="font-size:1.6rem;">{ai.get('emoji','')}</div>
                        <h2 style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#f0ede8;margin:0;">{acc_fav}</h2>
                        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{ai.get('warna','#888')};margin:0 0 0.8rem;">{ai.get('singkat','')}</p>
                        <p style="font-size:0.83rem;color:#777;line-height:1.7;">{ai.get('desc','')}</p>
                        <p style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#444;margin:0.8rem 0 0.4rem;">NICHE ←──→ BASIC</p>
                        <div style="display:flex;gap:0.3rem;">{skala}</div>
                    </div>""", unsafe_allow_html=True)

                # Top 3 mood
                # mood yang tidak masuk diberi keterangan akan memiliki keterangan wah hebat, keren
                st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:2rem 0 1rem;'>TOP 3 MOOD KAMU</p>", unsafe_allow_html=True)
                mc = st.columns(3)
                for i, (mood, count) in enumerate(top_mood):
                    mc[i].markdown(f"""
                    <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.1rem;">
                        <p style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#e8d5a3;margin:0;">#{i+1}</p>
                        <h3 style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#f0ede8;margin:0.2rem 0;text-transform:capitalize;">{mood}</h3>
                        <p style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#444;margin:0 0 0.5rem;">muncul {count}×</p>
                        <p style="font-size:0.78rem;color:#666;line-height:1.6;">{MOOD_DESC.get(mood,'wah hebat, keren.')}</p>
                    </div>""", unsafe_allow_html=True)

                # List rekomendasi album
                st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:2rem 0 1rem;'>{n_rek} REKOMENDASI ALBUM</p>", unsafe_allow_html=True)
                for i, (_, r) in enumerate(df_rek.iterrows()):
                    # Ambil 4 vibe pertama sbg preview deskriptor kecuali yang diexclude
                    vibes = ""
                    if pd.notna(r.get("descriptors")):
                        vibes = " · ".join([k.strip() for k in r["descriptors"].split(",")
                                            if k.strip() not in EXCLUDE][:4])
                    # Tombol Spotify cuma muncul kalau URL valid
                    spot = ""
                    url = r.get("spotify_search_url", "")
                    if pd.notna(url) and str(url).startswith("http"):
                        spot = f'<a href="{url}" target="_blank" style="display:inline-block;background:#1DB954;color:#000;font-family:DM Mono,monospace;font-size:0.6rem;font-weight:700;padding:0.3rem 0.8rem;border-radius:20px;text-decoration:none;margin-top:0.5rem;">▶ SPOTIFY</a>'

                    #css untuk tampilan font
                    st.markdown(f"""
                    <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1rem 1.3rem;margin-bottom:0.5rem;">
                        <span style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2a2a2a;">#{i+1}</span>
                        <span style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#f0ede8;margin-left:0.5rem;">{r['release_name']}</span>
                        <span style="font-size:0.78rem;color:#555;margin-left:0.5rem;">{r['artist_name']} · {int(r['year'])}</span>
                        <div style="display:flex;gap:1.5rem;margin-top:0.5rem;flex-wrap:wrap;font-size:0.75rem;color:#666;">
                            <span>★ <b style="color:#e8d5a3;">{r['avg_rating']:.2f}</b></span>
                            <span>{r['primary_genres']}</span>
                            <span>{r['acc_label']}</span>
                        </div>
                        {f'<p style="font-family:DM Mono,monospace;font-size:0.6rem;color:#333;margin:0.4rem 0 0;">{vibes}</p>' if vibes else ''}
                        {spot}
                    </div>""", unsafe_allow_html=True)



# HALAMAN 4: ABOUT ME
elif st.session_state.page == "about":

    st.markdown("""
    <div style="text-align:center;padding:3rem 0 2rem;">
        <img src="https://f4.bcbits.com/img/a0744100055_16.jpg" 
             style="border-radius:50%;border:2px solid #2a2a2a;width:180px;height:180px;object-fit:cover;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:2.5rem;">
        <h1 style="font-family:'DM Serif Display',serif;font-size:2.5rem;font-weight:400;color:#f0ede8;margin:0;">
            RIGEL AMADEUS VOLKER
        </h1>
        <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#444;letter-spacing:0.2em;margin-top:0.5rem;">
            SISWA · DATA SCIENCE · MUSIK NERD
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.5rem;margin-bottom:1rem;">
            <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:0 0 0.75rem;">TENTANG</p>
            <p style="font-size:0.88rem;color:#777;line-height:1.9;margin:0;">
                Nama saya Rigel Amadeus Volker, siswa SMKN Purblingga jurusan RPL kelas 11. Saya menyukai musik dan film
                    karena itu saya membuat projek machin learning ini.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.5rem;">
            <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:0 0 0.75rem;">KONTAK</p>
            <p style="font-size:0.85rem;color:#666;line-height:2;margin:0;">
                📧 rigel123@gmail.com<br>
                🐙 github.com/rigelgithub<br>
                🎵 last.fm/gapunya/gapunya123
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.5rem;margin-bottom:1rem;">
            <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:0 0 0.75rem;">TENTANG PROYEK</p>
            <p style="font-size:0.88rem;color:#777;line-height:1.9;margin:0;">
                REQ adalah sistem rekomendasi album berbasis Machine Learning 
                yang menggunakan dataset dari RateYourMusic. Dengan rekomendasi dan prediksi yang akurat
                    dan keren.
            </p>
        </div>
        """, unsafe_allow_html=True)

        #font
        st.markdown("""
        <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:4px;padding:1.5rem;">
            <p style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#444;letter-spacing:0.15em;margin:0 0 0.75rem;">ALAT YANG DIGUNAKAN</p>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">Python</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">Streamlit</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">scikit-learn</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">pandas</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">Plotly</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">joblib</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#e8d5a3;background:#1a1a1a;padding:0.25rem 0.6rem;border-radius:2px;border:1px solid #2a2a2a;">RateYourMusic</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# HALAMAN 5: KODE PROYEK
elif st.session_state.page == "kode":

    # Header halaman kode proyek
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:3rem;font-weight:400;color:#f0ede8;margin-bottom:0.3rem;">Kode Proyek</h1>
    <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#444;letter-spacing:0.15em;margin-bottom:2rem;">
        proyek_reyal.ipynb — ditampilkan cell per cell
    </p>
    """, unsafe_allow_html=True)

    # Render tiap cell notebook: markdown -> st.markdown, code -> st.code

    st.markdown('## 1. Import Library')

  #import kode jupyter notebook supaya bisa dibaca di streamlit
    st.code("""# pandas, manipulasi data tabel
import pandas as pd
#operasi matematis
import numpy as np
# matplotlib, visualisasi grafik
import matplotlib.pyplot as plt
# sklearn, semua keperluan machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Counter, menghitung frekuensi item
from collections import Counter
# joblib , menyimpan model
import joblib
# warnings, menyembunyikan pesan peringatan yang tidak penting
import warnings
warnings.filterwarnings('ignore')""", language="python")

    st.markdown('## 2. Load dan Eksplorasi Dataset')
    st.code("""df = pd.read_csv("rym.csv")
df.head()""", language="python")
    st.code("""df.shape""", language="python")
    st.code("""df.info()""", language="python")
    st.code("""df.describe()""", language="python")
    st.code("""df.isna().sum()""", language="python")

    st.markdown('## 3. Feature Engineering — Membuat Label Era & Aksesibilitas')
    st.code("""# FEATURE ENGINEERING: Membuat fitur baru dari data yang ada

# Ambil tahun dari tanggal rilis
df['year'] = pd.to_datetime(df['release_date']).dt.year

# Transformasi log untuk fitur numerik yang skewnya tinggi
df['log_rating_count'] = np.log1p(df['rating_count'])
df['log_review_count']  = np.log1p(df['review_count'])

# Label Era
ERA_BINS   = [0, 1969, 1984, 1999, 2012, 2026]
ERA_LABELS = ['Pionir', 'Old School', 'Mid High School', 'New School', 'New New School']
df['era'] = pd.cut(df['year'], bins=ERA_BINS, labels=ERA_LABELS)

# Label Aksesibilitas
ACC_BINS   = [0, 2, 4, 6, 8, 10.1]
ACC_LABELS = ['Niche', 'Elitis', 'Tidak Basic', 'Agak Lumayan Basic', 'Basic']
df['acc_label'] = pd.cut(df['accessibility'], bins=ACC_BINS, labels=ACC_LABELS)

# Hapus baris dengan NaN pada kolom label
df = df.dropna(subset=['era', 'acc_label'])

#liat jumlah data ditiap kategori
print(df['era'].value_counts().sort_index())
print(df['acc_label'].value_counts())""", language="python")
    st.code("""print("Distribusi Era:")
#hitung era dan urutkan berdasarkan urutan label
print(df['era'].value_counts().sort_index())
print()
print("Distribusi Aksesibilitas:")
print(df['acc_label'].value_counts())""", language="python")

    st.markdown('## 4. Visualisasi Distribusi')
    st.code("""# Visualisasi distribusi era dan aksesibilitas mdengan bar chart

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribusi Era
era_counts = df['era'].value_counts().sort_index()
axes[0].bar(era_counts.index, era_counts.values, color=['#2d6a4f','#52b788','#95d5b2','#b7e4c7','#d8f3dc'], edgecolor='black')
axes[0].set_title('Distribusi Album per Era', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Era')
axes[0].set_ylabel('Jumlah Album')
axes[0].tick_params(axis='x', rotation=20)
for i, v in enumerate(era_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=11)

# Distribusi Aksesibilitas

#urutab
acc_order = ['Niche', 'Elitis', 'Tidak Basic', 'Agak Lumayan Basic', 'Basic']
acc_counts = df['acc_label'].value_counts().reindex(acc_order)
#mengatur bar chart
axes[1].bar(acc_counts.index, acc_counts.values, color=['#03045e','#023e8a','#0077b6','#48cae4','#ade8f4'], edgecolor='black')
axes[1].set_title('Distribusi Album per Level Aksesibilitas', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Level Aksesibilitas')
axes[1].set_ylabel('Jumlah Album')
axes[1].tick_params(axis='x', rotation=20)
for i, v in enumerate(acc_counts.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontsize=11)

plt.tight_layout()
plt.show()""", language="python")

    st.markdown('## 5. Model 1: Klasifikasi Era\nuntuk prediksi era\n\nFitur yang digunakan: `year`, `avg_rating`, `log_rating_count`, `log_review_count`, `accessibility`')
    st.code("""#persiapan data
# X/fiturnya, year, rata2 rating, log jumlah penilai, log jumlah reiewer, skor aksesibilitas
# y/target, era

#memisah data latihan dan data tes
X_era = df[['year', 'avg_rating', 'log_rating_count', 'log_review_count', 'accessibility']]
y_era = df['era']

X_era_train, X_era_test, y_era_train, y_era_test = train_test_split(
    X_era, y_era, test_size=0.2, random_state=42
)""", language="python")
    st.code("""# melatih 3 model dan bandingkan akurasi
#   pipeline untuk membugkus model, StandardScaler, normalisasi fitur
era_models = { 
# Model 1 – Logistic Regression:
#   cepat dan mudah diinterpretasikan.
#   max_iter=1000 supaya konvergen pada dataset yang lebih besar.
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    
# Model 2 – Random Forest:
#   Ensemble dari banyak Decision Tree. Sangat kuat untuk data tabular.
#   n_estimators=100, membangun 100 pohon keputusan.
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    
# Model 3 – K-Nearest Neighbors (KNN):
#   Algoritma berbasis jarak, prediksi berdasarkan 5 tetangga terdekat.
#   n_neighbors=5, melihat 5 album paling mirip untuk menentukan era.
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
}

print("        HASIL PERBANDINGAN ERA MODEL")

hasil_era = {}
for nama, model in era_models.items():
    # Latih model
    model.fit(X_era_train, y_era_train)
    # Prediksi pada data uji
    y_pred = model.predict(X_era_test)
    # Hitung akurasi
    akurasi = accuracy_score(y_era_test, y_pred)
    hasil_era[nama] = akurasi
    print(f"  {nama:25s} → Akurasi: {akurasi*100:.2f}%")

#model yang terbaik dianu jadi bisa dipanggil nanti jika diperlukan
model_era_gacor = max(hasil_era, key=hasil_era.get)""", language="python")
    st.code("""#tes akurasi
model_era_gacor = era_models[model_era_gacor]
y_pred_era = model_era_gacor.predict(X_era_test)

#perlihatkan hasil
print(f"Classification Report – {model_era_gacor} (Era Model)")
print("-" * 55)
print(classification_report(y_era_test, y_pred_era))""", language="python")
    st.code("""# Perbandingan akurasi 3 model era dalam bentuk bar chart

plt.figure(figsize=(8, 5))
bars = plt.bar(hasil_era.keys(), [v * 100 for v in hasil_era.values()],
               color=['#4cc9f0', '#f72585', '#7209b7'], edgecolor='black')
plt.ylim(0, 110)
plt.ylabel('Akurasi (%)')
plt.title('Perbandingan Akurasi 3 Model – Klasifikasi Era', fontweight='bold')
for bar, val in zip(bars, hasil_era.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""", language="python")
    st.code("""#simpan menjadi file joblib
joblib.dump(model_era_gacor, "model_era_gacor.joblib")""", language="python")

    st.markdown('## 6. Model 2: Klasifikasi Aksesibilitas\n**Tujuan:** Memprediksi level aksesibilitas album\nkodenya hampir sama dengan model era\n\nFitur yang digunakan: `accessibility`, `avg_rating`, `log_rating_count`, `log_review_count`')
    st.code("""X_acc = df[['accessibility', 'avg_rating', 'log_rating_count', 'log_review_count']]
y_acc = df['acc_label']

X_acc_train, X_acc_test, y_acc_train, y_acc_test = train_test_split(
    X_acc, y_acc, test_size=0.2, random_state=42
)""", language="python")
    st.code("""acc_models = {
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
    y_pred = model.predict(X_acc_test)
    akurasi = accuracy_score(y_acc_test, y_pred)
    hasil_acc[nama] = akurasi
    print(f"  {nama:25s} Akurasi: {akurasi*100:.2f}%")
    
model_acc_gacor = max(hasil_acc, key=hasil_acc.get)""", language="python")
    st.code("""#laporan model terbaik
model_acc_gacor = acc_models[model_acc_gacor]
y_pred_acc =model_acc_gacor.predict(X_acc_test)

print(f"Classification Report – {model_acc_gacor} (Aksesibilitas Model)")
print("-" * 58)
print(classification_report(y_acc_test, y_pred_acc))""", language="python")
    st.code("""#perbandingan dengan bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(hasil_acc.keys(), [v * 100 for v in hasil_acc.values()],
               color=['#06d6a0', '#ef476f', '#ffd166'], edgecolor='black')
plt.ylim(0, 110)
plt.ylabel('Akurasi (%)')
plt.title('Perbandingan Akurasi 3 Model – Klasifikasi Aksesibilitas', fontweight='bold')
for bar, val in zip(bars, hasil_acc.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""", language="python")
    st.code("""#simpan menjadi joblib
joblib.dump(model_acc_gacor, "model_acc_gacor.joblib")""", language="python")

    st.markdown('## 7. Fungsi-Fungsi Utama Pipeline')
    st.code("""# FUNGSI 1: cari_album_di_dataset
# Menerima list album+artis dari pengguna, lalu mencari
# kecocokan di dataset RYM secara case-insensitive.
# Cara kerja:, Input diubah ke huruf kecil, Setiap baris dataset dicek: apakah nama album DAN artis,
#mengandung kata kunci yang dimasukkan?, Mengembalikan DataFrame baris-baris yang cocok
# Parameter:
#   input_list, list of dict, setiap dict punya key album dan artis
#   dataset, DataFrame RYM yang sudah diproses

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
        # Jika ketemu, ambil baris pertama (paling relevan)
        if not cocok.empty:
            hasil.append(cocok.iloc[0])
    
    if not hasil:
        return pd.DataFrame()
    return pd.DataFrame(hasil)""", language="python")
    st.code("""# FUNGSI 2: prediksi_era_pengguna
# Dari album-album yang ditemukan di dataset, gunakan model ERA
# untuk memprediksi era setiap album, lalu tentukan era yang
# paling sering muncul = era favorit pengguna.
# Cara kerja:, Ambil fitur dari setiap album yang ditemukan, Minta model era memprediksi era album tersebut,
#Counter untuk menghitung era mana yang paling dominan, Jika seri, ambil era yang rata-rata ratingnya lebih tinggi

ERA_KETERANGAN = {
    'Pionir': {
        'rentang'    : '~1920 - 1969',
        'keterangan' : 'Orang tuwa. jaman jamannya bob dylan, black sabbath, fleetwood mac dll'
    },
    'Old School': {
        'rentang'    : '1970 - 1984',
        'keterangan' : 'era keemasan rock(mungkin). pink floyd, king crimson, yes'
    },
    'Mid High School': {
        'rentang'    : '1985 - 1999',
        'keterangan' : 'eranya musik yang kasar seperti grunge. jamannya nirvana dll'
    },
    'New School': {
        'rentang'    : '2000 - 2012',
        'keterangan' : 'masuk era modern dan digital, musik elektronik dan emo makin membabi buta. LCD soundsystem, my chemical romance, greenday dll.'
    },
    'New New School': {
        'rentang'    : '2013 - sekarang',
        'keterangan' : 'Era streaming, era palng eksperimental, bervariasi dan tidak jelas. salah satu era favorit saya. kendrick lamar, laufey, deathgrips.'
    }
}

def prediksi_era_pengguna(df_album_cocok, model_era):
    fitur = df_album_cocok[['year', 'avg_rating', 'log_rating_count', 'log_review_count', 'accessibility']]
    prediksi = model_era.predict(fitur)
    # Hitung era yang paling sering muncul
    era_dominan = Counter(prediksi).most_common(1)[0][0]
    return era_dominan, prediksi""", language="python")
    st.code("""# FUNGSI 3: prediksi_aksesibilitas_pengguna
# Dari album yang ditemukan, prediksi level aksesibilitas tiap album
# lalu cari yang paling dominan = selera aksesibilitas pengguna.

ACC_KETERANGAN = {
    'Niche': {
        'emoji'      : '🔬',
        'keterangan' : 'Selera kamu sangat niche, bisa dipastikan tidak pernah diberi akses speaker. jarang mandi.'
    },
    'Elitis': {
        'emoji'      : '🎓',
        'keterangan' : 'Selera kamu elitis, bisa dipastikan jarang diberi akses speaker.'
    },
    'Tidak Basic': {
        'emoji'      : '🎸',
        'keterangan' : 'Selera kamu tidak basic, berpendirian dan berpendidikan dan punya kemampuan bersosialisasi.'
    },
    'Agak Lumayan Basic': {
        'emoji'      : '🎧',
        'keterangan' : 'selerammu agak lumayan basic, ehh, bolelahbolelah.'
    },
    'Basic': {
        'emoji'      : '📻',
        'keterangan' : 'selera kamu mainstream yang berarti kamu orangnya basic dan minim literasi, FOMO dan rambutnya modelan dirapiin dikit aja pak.'
    }
}

def prediksi_aksesibilitas_pengguna(df_album_cocok, model_acc):
    fitur = df_album_cocok[['accessibility', 'avg_rating', 'log_rating_count', 'log_review_count']]
    prediksi = model_acc.predict(fitur)
    acc_dominan = Counter(prediksi).most_common(1)[0][0]
    return acc_dominan, prediksi""", language="python")
    st.code("""# FUNGSI 4: analisis_mood
# Menganalisis deskriptor dari semua album yang ditemukan lalu mengembalikan 3 mood teratas.
# Cara kerja:
# Pisahkan setiap string descriptor menjadi list kata, Kumpulkan semua kata dari semua album ke satu list besar
#Gunakan Counter untuk menghitung frekuensi masing-masing, Kembalikan 3 yang paling sering muncul

# Beberapa kata di-exclude karena bersifat teknis
EXCLUDE_DESCRIPTOR = {'malevocals', 'femalevocals', 'conceptalbum', 'instrumental'}

def analisis_mood(df_album_cocok):
    semua_descriptor = []
    for desc_str in df_album_cocok['descriptors'].dropna():
        kata_kata = [k.strip() for k in desc_str.split(',')]
        kata_kata = [k for k in kata_kata if k not in EXCLUDE_DESCRIPTOR]
        semua_descriptor.extend(kata_kata)
    
    if not semua_descriptor:
        return [('tidak tersedia', 0)]
    
    top3 = Counter(semua_descriptor).most_common(3)
    return top3""", language="python")
    st.code("""# FUNGSI 5: rekomendasikan_album
# Merekomendasikan 5 album dari dataset yang kemungkinan disukai
# pengguna, berdasarkan: Era yang sama dengan era favorit pengguna, Level aksesibilitas yang sama atau berdekatan, Descriptor yang paling banyak overlap dengan selera
# Cara kerja : Filter dataset berdasarkan era favorit, yang cocok dengan top mood pengguna dll

ACC_ORDER = ['Niche', 'Elitis', 'Tidak Basic', 'Agak Lumayan Basic', 'Basic']

def rekomendasikan_album(era_favorit, acc_favorit, top_mood, df_album_cocok, dataset):
    # Kumpulkan nama album yang sudah disukai (untuk exclude)
    nama_sudah_suka = set(df_album_cocok['release_name'].str.lower())
    
    # cari yang nilai aksesibilitasnya mirip degan sipengguna
    acc_idx = ACC_ORDER.index(acc_favorit) if acc_favorit in ACC_ORDER else 2
    acc_range_idx = list(set([max(0, acc_idx - 1), acc_idx, min(4, acc_idx + 1)]))
    acc_range = [ACC_ORDER[i] for i in acc_range_idx]
    
    # Kumpulkan kata mood teratas pengguna
    kata_mood = set([m[0] for m in top_mood])
    
    # Filter berdasarkan era dan aksesibilitas
    kandidat = dataset[
        (dataset['era'] == era_favorit) &
        (dataset['acc_label'].isin(acc_range)) &
        (~dataset['release_name'].str.lower().isin(nama_sudah_suka))
    ].copy()
    
    # Jika terlalu sedikit kandidat, buka ke semua aksesibilitas era yang sama
    if len(kandidat) < 5:
        kandidat = dataset[
            (dataset['era'] == era_favorit) &
            (~dataset['release_name'].str.lower().isin(nama_sudah_suka))
        ].copy()
    
    # Hitung skor mood yang sama dengan sipengguna
    def hitung_skor_mood(desc_str):
        if pd.isna(desc_str):
            return 0
        kata = set([k.strip() for k in desc_str.split(',')])
        return len(kata & kata_mood)
    
    kandidat['skor_mood'] = kandidat['descriptors'].apply(hitung_skor_mood)
    
    # Urutkan: prioritas skor mood, lalu avg_rating
    kandidat = kandidat.sort_values(
        by=['skor_mood', 'avg_rating'], ascending=[False, False]
    )
    
    # Pilih kolom yang ditampilkan
    kolom_tampil = ['release_name', 'artist_name', 'year', 'primary_genres',
                    'avg_rating', 'acc_label', 'spotify_search_url']
    
    return kandidat[kolom_tampil].head(5)""", language="python")

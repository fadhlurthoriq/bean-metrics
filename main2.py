import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bean Metrics",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# DARK THEME CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0E0E0F;
    --bg2:      #141416;
    --surface:  #1A1A1D;
    --surface2: #222226;
    --border:   #2A2A2F;
    --border2:  #333338;
    --text:     #F0F0F2;
    --text2:    #8A8A95;
    --text3:    #555560;
    --accent:   #C8A97E;
    --green:    #4ADE80;
    --radius:   14px;
    --radius-s: 8px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.block-container {
    padding: 0 2.5rem 5rem 2.5rem !important;
    max-width: 1140px !important;
}

/* ===== NAVBAR ===== */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 0 0;
    height: 62px;
    background: rgba(14,14,15,0.9);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    margin: 0 0 3rem 0;
}
.nav-brand { display: flex; align-items: center; gap: 0.75rem; }
.nav-logo {
    width: 38px; height: 38px;
    background: var(--text);
    border-radius: 10px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.6rem; font-weight: 800;
    color: var(--bg); line-height: 1.2;
    letter-spacing: -0.01em;
}
.nav-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.01em;
    display: block;
}
.nav-sub { font-size: 0.7rem; color: var(--text2); display: block; }
.nav-center { font-size: 0.85rem; color: var(--text2); font-style: italic; }
.nav-right { display: flex; align-items: center; gap: 0.6rem; }
.nav-pill {
    padding: 0.3rem 0.8rem;
    border: 1px solid var(--border2);
    border-radius: 100px;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--text2);
}
.nav-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px rgba(74,222,128,0.6);
}

/* ===== HERO ===== */
.hero {
    padding: 1.5rem 0 3rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 3rem;
}
.hero-eyebrow {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    line-height: 1.05; letter-spacing: -0.03em;
    color: var(--text); margin-bottom: 1rem;
}
.hero h1 span { color: var(--accent); }
.hero p {
    font-size: 0.98rem; color: var(--text2);
    line-height: 1.65; max-width: 540px;
}

/* ===== METRICS ===== */
.metric-strip {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden; margin-bottom: 3rem;
}
.metric-item { background: var(--surface); padding: 1.5rem; }
.m-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: var(--text); display: block;
    letter-spacing: -0.03em;
}
.m-lab {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text3); display: block; margin-top: 0.3rem;
}

/* ===== SECTION LABEL ===== */
.section-label {
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.sl-num { color: var(--accent); }
.section-label::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ===== FORM CARD ===== */
.form-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem; margin-bottom: 1.25rem;
}

/* ===== SELECTBOX DARK ===== */
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-s) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
div[data-baseweb="select"] svg { fill: var(--text2) !important; }
[data-baseweb="popover"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-s) !important;
}
[role="option"] { background: var(--surface2) !important; color: var(--text) !important; }
[role="option"]:hover, [aria-selected="true"] {
    background: var(--surface) !important; color: var(--accent) !important;
}
.stSelectbox label, label {
    font-size: 0.73rem !important; font-weight: 600 !important;
    color: var(--text2) !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: var(--text) !important; color: var(--bg) !important;
    border: none !important; border-radius: var(--radius-s) !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    letter-spacing: 0.03em !important; width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent) !important; color: var(--bg) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(200,169,126,0.25) !important;
}

/* ===== INFO BOX ===== */
.info-box {
    background: rgba(200,169,126,0.07);
    border: 1px solid rgba(200,169,126,0.18);
    border-radius: var(--radius-s);
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem; color: var(--accent);
    margin-top: 1rem;
}

/* ===== RESULT CARDS ===== */
.result-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.75rem;
    transition: transform 0.2s, border-color 0.2s;
}
.result-card:hover { transform: translateY(-3px); border-color: var(--border2); }
.result-card.best { background: var(--text); border-color: var(--text); }
.result-card.best .rc-badge { background: #1A1A1D; color: var(--accent); }
.result-card.best .rc-price,
.result-card.best .rc-value { color: #0E0E0F; }
.result-card.best .rc-label,
.result-card.best .rc-sub { color: #888; }
.result-card.best .rc-divider { background: #D8D8D8; }

.rc-badge {
    display: inline-block; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.28rem 0.65rem; border-radius: 100px;
    background: var(--surface2); color: var(--accent);
    margin-bottom: 1.25rem;
}
.rc-price {
    font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.03em; line-height: 1;
    margin-bottom: 0.2rem;
}
.rc-sub { font-size: 0.7rem; color: var(--text3); margin-bottom: 0.4rem; }
.rc-divider { height: 1px; background: var(--border); margin: 1rem 0; }
.rc-label {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text3); margin-bottom: 0.2rem;
}
.rc-value { font-size: 0.97rem; font-weight: 600; color: var(--text); margin-bottom: 0.75rem; }

/* ===== SUMMARY ===== */
.summary-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1.75rem;
}
.s-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.6rem;
}
.s-text { font-size: 0.97rem; color: var(--text2); line-height: 1.7; }
.s-text strong { color: var(--text); }

.stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }
[data-testid="stForm"] { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_excel("data/coffee_shop.xlsx")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['hour']      = df['transaction_time'].dt.hour
    df['month']     = df['transaction_date'].dt.month
    df['dayofweek'] = df['transaction_date'].dt.dayofweek
    df = df.dropna(subset=['transaction_qty','unit_price','product_category',
                           'product_type','city_location','subdistrict_name'])
    df_ml = df.copy()
    le_cat = LabelEncoder(); le_type = LabelEncoder()
    le_city = LabelEncoder(); le_sub = LabelEncoder()
    df_ml['product_category'] = le_cat.fit_transform(df_ml['product_category'])
    df_ml['product_type']     = le_type.fit_transform(df_ml['product_type'])
    df_ml['city_location']    = le_city.fit_transform(df_ml['city_location'])
    df_ml['subdistrict_name'] = le_sub.fit_transform(df_ml['subdistrict_name'])
    X = df_ml[['city_location','subdistrict_name','product_category',
               'product_type','unit_price','hour','month','dayofweek']]
    y = df_ml['transaction_qty']
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(X, y)
    return mdl, df, le_cat, le_type, le_city, le_sub, X.columns.tolist()


def predict(mdl, le_cat, le_type, le_city, le_sub, df, cols,
            kategori, tipe, city, subdistrict):
    subset = df[(df['product_category']==kategori)&(df['product_type']==tipe)&(df['city_location']==city)]
    if len(subset)<5: subset = df[(df['product_category']==kategori)&(df['product_type']==tipe)]
    if len(subset)<5: subset = df[df['product_category']==kategori]
    if len(subset)<5: subset = df
    hm = subset['unit_price'].mean()
    hasil = []
    for h in [hm*0.85, hm, hm*1.15]:
        row = pd.DataFrame([[le_city.transform([city])[0], le_sub.transform([subdistrict])[0],
                             le_cat.transform([kategori])[0], le_type.transform([tipe])[0],
                             h, 12, 6, 2]], columns=cols)
        qty = mdl.predict(row)[0] * 30
        hasil.append((h, qty, h*qty))
    return sorted(hasil, key=lambda x: x[2], reverse=True)


# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">B<br>M</div>
    <div>
      <span class="nav-name">Bean Metrics</span>
      <span class="nav-sub">Price Intelligence</span>
    </div>
  </div>
  <div class="nav-center">Hello, i'm Beans ☕</div>
  <div class="nav-right">
    <div class="nav-dot"></div>
    <div class="nav-pill">Model Active</div>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">☕ Coffee Shop · Machine Learning</div>
  <h1>Prediksi <span>Harga</span> &<br>Penjualan Produk</h1>
  <p>Temukan harga optimal dan estimasi omset bulanan berdasarkan data transaksi nyata menggunakan algoritma Random Forest.</p>
</div>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
with st.spinner("Melatih model..."):
    try:
        mdl, df, le_cat, le_type, le_city, le_sub, cols = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {e}")
        st.info("Pastikan file `data/coffee_shop.xlsx` ada di folder `data/`.")
        model_loaded = False

if model_loaded:
    # ---- METRICS ----
    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-item">
            <span class="m-val">{len(df):,}</span>
            <span class="m-lab">Total Transaksi</span>
        </div>
        <div class="metric-item">
            <span class="m-val">{df['city_location'].nunique()}</span>
            <span class="m-lab">Kota</span>
        </div>
        <div class="metric-item">
            <span class="m-val">{df['product_type'].nunique()}</span>
            <span class="m-lab">Jenis Produk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- FORM ----
    st.markdown('<div class="section-label"><span class="sl-num">01</span> — Pilih Produk & Lokasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        kategori = st.selectbox("Kategori Produk", sorted(df['product_category'].unique()))
        tipe = st.selectbox("Jenis Produk", sorted(df[df['product_category']==kategori]['product_type'].unique()))
    with col2:
        city = st.selectbox("Kota", sorted(df['city_location'].unique()))
        subdistrict = st.selectbox("Kecamatan", sorted(df[df['city_location']==city]['subdistrict_name'].unique()))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 Model menghasilkan <strong>3 skenario harga</strong>: Diskon (–15%), Normal, dan Premium (+15%) beserta estimasi omset bulanan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run = st.button("Prediksi Sekarang →")

    # ---- HASIL ----
    if run:
        with st.spinner("Menghitung prediksi..."):
            hasil = predict(mdl, le_cat, le_type, le_city, le_sub,
                            df, cols, kategori, tipe, city, subdistrict)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="sl-num">02</span> — Hasil Prediksi</div>', unsafe_allow_html=True)

        labels = [("PREMIUM",""),("SEIMBANG","best"),("DISKON","")]
        icons  = ["💎","⚖️","🔥"]

        cr = st.columns(3, gap="medium")
        for i, ((h, qty, omzet),(lab,cls),icon) in enumerate(zip(hasil, labels, icons)):
            h_idr = int(h*17000); q_int = int(qty); o_idr = int(omzet*17000)
            with cr[i]:
                st.markdown(f"""
                <div class="result-card {cls}">
                    <span class="rc-badge">{icon} {lab}</span>
                    <div class="rc-price">Rp {h_idr:,}</div>
                    <div class="rc-sub">per cup</div>
                    <div class="rc-divider"></div>
                    <div class="rc-label">Estimasi Terjual</div>
                    <div class="rc-value">{q_int:,} cup / bulan</div>
                    <div class="rc-label">Estimasi Omset</div>
                    <div class="rc-value">Rp {o_idr:,}</div>
                </div>
                """, unsafe_allow_html=True)

        best_h = int(hasil[1][0]*17000); best_o = int(hasil[1][2]*17000)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="sl-num">03</span> — Ringkasan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-card">
            <div class="s-label">Rekomendasi · {tipe} · {city}, {subdistrict}</div>
            <div class="s-text">
                Harga optimal untuk <strong>{tipe}</strong> di <strong>{city} ({subdistrict})</strong>
                adalah <strong>Rp {best_h:,}</strong> per cup, dengan estimasi omset bulanan
                sebesar <strong>Rp {best_o:,}</strong>. Pertimbangkan skenario <strong>Premium</strong>
                atau <strong>Diskon</strong> untuk menyesuaikan segmen pasar yang dituju.
            </div>
        </div>
        """, unsafe_allow_html=True)
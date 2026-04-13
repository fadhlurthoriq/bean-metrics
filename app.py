import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bean Metrics — Prediksi Harga & Penjualan",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CSS — DARK + MOBILE FIRST
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ===== TOKENS ===== */
:root {
    --bg:       #0E0E0F;
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

/* ===== BASE ===== */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Mobile-first padding — compact on small screens */
.block-container {
    padding: 0 1rem 4rem 1rem !important;
    max-width: 1140px !important;
}

/* ===== NAVBAR ===== */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    background: rgba(14,14,15,0.95);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    margin: 0 -1rem 2rem -1rem;
    padding: 0 1rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.nav-brand { display: flex; align-items: center; gap: 0.6rem; }
.nav-logo {
    width: 34px; height: 34px;
    background: var(--text);
    border-radius: 8px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.55rem; font-weight: 800;
    color: var(--bg); line-height: 1.2;
}
.nav-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem; font-weight: 700;
    color: var(--text); display: block;
}
.nav-sub {
    font-size: 0.62rem; color: var(--text2); display: block;
}
/* Hide center tagline on mobile, show on tablet+ */
.nav-center {
    font-size: 0.82rem; color: var(--text2); font-style: italic;
    display: none;
}
.nav-right { display: flex; align-items: center; gap: 0.5rem; }
.nav-pill {
    padding: 0.25rem 0.65rem;
    border: 1px solid var(--border2);
    border-radius: 100px;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text2);
}
.nav-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px rgba(74,222,128,0.6);
}

/* ===== HERO ===== */
.hero {
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    line-height: 1.1; letter-spacing: -0.02em;
    color: var(--text); margin-bottom: 0.75rem;
}
.hero h1 span { color: var(--accent); }
.hero p {
    font-size: 0.9rem; color: var(--text2);
    line-height: 1.6; max-width: 100%;
}

/* ===== METRIC STRIP ===== */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 2rem;
}
.metric-item { background: var(--surface); padding: 1rem 0.75rem; }
.m-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 700;
    color: var(--text); display: block;
    letter-spacing: -0.02em;
}
.m-lab {
    font-size: 0.58rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text3); display: block; margin-top: 0.2rem;
}

/* ===== SECTION LABEL ===== */
.section-label {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text3); margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
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
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* ===== SELECTBOX DARK ===== */
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-s) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    min-height: 48px !important;  /* Bigger tap target on mobile */
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover { border-color: var(--accent) !important; }
div[data-baseweb="select"] svg { fill: var(--text2) !important; }
[data-baseweb="popover"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-s) !important;
}
[role="option"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    min-height: 44px !important;  /* Bigger tap targets */
    display: flex !important;
    align-items: center !important;
}
[role="option"]:hover, [aria-selected="true"] {
    background: var(--surface) !important; color: var(--accent) !important;
}
.stSelectbox label, label {
    font-size: 0.72rem !important; font-weight: 600 !important;
    color: var(--text2) !important; letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: var(--text) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: var(--radius-s) !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    min-height: 52px !important;  /* Easy to tap on mobile */
    transition: all 0.2s !important;
    touch-action: manipulation !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 6px 20px rgba(200,169,126,0.25) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;  /* Tactile feedback on mobile tap */
}

/* ===== INFO BOX ===== */
.info-box {
    background: rgba(200,169,126,0.07);
    border: 1px solid rgba(200,169,126,0.18);
    border-radius: var(--radius-s);
    padding: 0.85rem 1rem;
    font-size: 0.82rem; color: var(--accent);
    margin-top: 0.75rem;
    line-height: 1.5;
}

/* ===== RESULT CARDS — MOBILE STACK ===== */
/* On mobile: cards are full width stacked vertically via HTML grid */
.result-grid-mobile {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    transition: border-color 0.2s;
}
.result-card.best {
    background: var(--text);
    border-color: var(--text);
}
.result-card.best .rc-badge { background: #1A1A1D; color: var(--accent); }
.result-card.best .rc-price,
.result-card.best .rc-value { color: #0E0E0F; }
.result-card.best .rc-label,
.result-card.best .rc-sub { color: #888; }
.result-card.best .rc-divider { background: #D8D8D8; }

/* Horizontal layout inside each card on mobile */
.rc-inner {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.rc-left { flex: 1; min-width: 0; }
.rc-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
    text-align: right;
    flex-shrink: 0;
}

.rc-badge {
    display: inline-block; font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.25rem 0.6rem; border-radius: 100px;
    background: var(--surface2); color: var(--accent);
    margin-bottom: 0.5rem;
}
.rc-price {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem; font-weight: 700;
    color: var(--text); letter-spacing: -0.02em; line-height: 1;
    margin-bottom: 0.15rem;
}
.rc-sub { font-size: 0.68rem; color: var(--text3); }
.rc-divider { height: 1px; background: var(--border); margin: 0.75rem 0; }
.rc-label {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text3); margin-bottom: 0.15rem;
}
.rc-value { font-size: 0.92rem; font-weight: 600; color: var(--text); }

/* ===== SUMMARY CARD ===== */
.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 1.25rem;
}
.s-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.5rem;
}
.s-text { font-size: 0.92rem; color: var(--text2); line-height: 1.65; }
.s-text strong { color: var(--text); }

/* ===== SPINNER ===== */
.stSpinner > div {
    border-color: var(--accent) transparent transparent transparent !important;
}
[data-testid="stForm"] { border: none !important; padding: 0 !important; }

/* ===========================
   RESPONSIVE BREAKPOINTS
   =========================== */

/* Tablet and up (≥ 640px) */
@media (min-width: 640px) {
    .block-container {
        padding: 0 1.75rem 5rem 1.75rem !important;
    }
    .navbar {
        margin: 0 -1.75rem 2.5rem -1.75rem;
        padding: 0 1.75rem;
        height: 60px;
    }
    .nav-center { display: block; }
    .hero h1 { font-size: 2.8rem; }
    .hero { margin-bottom: 2.5rem; padding-bottom: 2.5rem; }
    .m-val { font-size: 1.75rem; }
    .m-lab { font-size: 0.63rem; }
    .metric-item { padding: 1.25rem; }
    .form-card { padding: 1.75rem; }
    .result-card { padding: 1.5rem; }

    /* On tablet+: switch result cards to horizontal row */
    .result-grid-mobile {
        flex-direction: row;
        gap: 0.85rem;
    }
    .result-grid-mobile .result-card { flex: 1; }

    /* Reset inner card to vertical layout on tablet+ */
    .rc-inner { flex-direction: column; gap: 0; }
    .rc-right { align-items: flex-start; text-align: left; }
    .rc-badge { margin-bottom: 0.85rem; }
    .rc-price { font-size: 1.75rem; }
}

/* Desktop (≥ 1024px) */
@media (min-width: 1024px) {
    .block-container {
        padding: 0 2.5rem 5rem 2.5rem !important;
    }
    .navbar {
        margin: 0 -2.5rem 3rem -2.5rem;
        padding: 0 2.5rem;
    }
    .hero h1 { font-size: 3.2rem; }
    .hero { margin-bottom: 3rem; }
    .m-val { font-size: 2rem; }
    .metric-item { padding: 1.5rem; }
    .result-card { padding: 1.75rem; }
    .rc-price { font-size: 1.9rem; }
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_resource(show_spinner=False)
def load_model():
    df = pd.read_excel("data/drinks_data.xlsx")
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
            <span class="m-lab">Transaksi</span>
        </div>
        <div class="metric-item">
            <span class="m-val">{df['city_location'].nunique()}</span>
            <span class="m-lab">Kota</span>
        </div>
        <div class="metric-item">
            <span class="m-val">{df['product_type'].nunique()}</span>
            <span class="m-lab">Produk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- FORM ----
    st.markdown('<div class="section-label"><span class="sl-num">01</span> — Pilih Produk & Lokasi</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    # Di mobile: 1 kolom. Di desktop: 2 kolom.
    # Streamlit tidak bisa deteksi screen size, jadi kita pakai 1 kolom
    # supaya konsisten dan readable di semua ukuran layar.
    kategori = st.selectbox("Kategori Produk", sorted(df['product_category'].unique()))
    tipe = st.selectbox("Jenis Produk", sorted(df[df['product_category']==kategori]['product_type'].unique()))
    city = st.selectbox("Kota", sorted(df['city_location'].unique()))
    subdistrict = st.selectbox("Kecamatan", sorted(df[df['city_location']==city]['subdistrict_name'].unique()))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 Model menghasilkan <strong>3 skenario harga</strong>: Diskon (–15%), Normal, dan Premium (+15%)
        beserta estimasi omset bulanan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Button full width — mudah di-tap di HP
    run = st.button("Prediksi Sekarang →")

    # ---- HASIL ----
    if run:
        with st.spinner("Menghitung prediksi..."):
            hasil = predict(mdl, le_cat, le_type, le_city, le_sub,
                            df, cols, kategori, tipe, city, subdistrict)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="sl-num">02</span> — Hasil Prediksi</div>',
                    unsafe_allow_html=True)

        labels = [("TERBAIK", "best"), ("SEIMBANG", ""), ("PREMIUM", "")]
        icons  = ["🔥", "⚖️", "💎"]

        # Render semua kartu dalam satu div — CSS yang atur layout-nya
        # (stack di mobile, row di tablet+)
        cards_html = '<div class="result-grid-mobile">'
        for i, ((h, qty, omzet), (lab, cls), icon) in enumerate(zip(hasil, labels, icons)):
            h_idr = int(h * 1000)
            q_int = int(qty)
            o_idr = int(omzet * 1000)
            cards_html += f"""
            <div class="result-card {cls}">
                <div class="rc-inner">
                    <div class="rc-left">
                        <span class="rc-badge">{icon} {lab}</span>
                        <div class="rc-price">Rp {h_idr:,}</div>
                        <div class="rc-sub">per cup</div>
                    </div>
                    <div class="rc-right">
                        <div>
                            <div class="rc-label">Terjual</div>
                            <div class="rc-value">{q_int:,} cup/bln</div>
                        </div>
                        <div>
                            <div class="rc-label">Omset</div>
                            <div class="rc-value">Rp {o_idr:,}</div>
                        </div>
                    </div>
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ---- SUMMARY ----
        best_h = int(hasil[0][0] * 1000)
        best_o = int(hasil[0][2] * 1000)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="sl-num">03</span> — Ringkasan</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="summary-card">
            <div class="s-label">Rekomendasi · {tipe} · {city}, {subdistrict}</div>
            <div class="s-text">
                Harga optimal untuk <strong>{tipe}</strong> di <strong>{city} ({subdistrict})</strong>
                adalah <strong>Rp {best_h:,}</strong> per cup, dengan estimasi omset bulanan
                sebesar <strong>Rp {best_o:,}</strong>. Pertimbangkan skenario <strong>Seimbang</strong>
                atau <strong>Premium</strong> untuk menyesuaikan segmen pasar yang dituju.
                Ukuran per cup adalah standard 250ml untuk semua varian.
            </div>
        </div>
        """, unsafe_allow_html=True)
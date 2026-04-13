import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# LOAD DATA
df = pd.read_excel("data/drinks_data.xlsx")

# bersihin nama kolom (WAJIB)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# FUNCTION PILIH OPSI
def pilih_opsi(nama, opsi):
    print(f"\nPilih {nama}:")
    for i, val in enumerate(opsi):
        print(f"{i+1}. {val}")
    
    pilihan = int(input("Masukkan nomor: ")) - 1
    return opsi[pilihan]

# PREPROCESSING
df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

df['hour'] = df['transaction_time'].dt.hour
df['month'] = df['transaction_date'].dt.month
df['dayofweek'] = df['transaction_date'].dt.dayofweek

df = df.dropna(subset=[
    'transaction_qty',
    'unit_price',
    'product_category',
    'product_type',
    'city_location',
    'subdistrict_name'
])

# COPY UNTUK ML
df_ml = df.copy()

# ENCODING
le_category = LabelEncoder()
le_type = LabelEncoder()
le_city = LabelEncoder()
le_subdistrict = LabelEncoder()

df_ml['product_category'] = le_category.fit_transform(df_ml['product_category'])
df_ml['product_type'] = le_type.fit_transform(df_ml['product_type'])
df_ml['city_location'] = le_city.fit_transform(df_ml['city_location'])
df_ml['subdistrict_name'] = le_subdistrict.fit_transform(df_ml['subdistrict_name'])

# FEATURE & TARGET
X = df_ml[['city_location', 'subdistrict_name', 'product_category', 'product_type', 'unit_price', 'hour', 'month', 'dayofweek']]
y = df_ml['transaction_qty']

# MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("\n✅ Model siap digunakan!\n")

# INPUT USER

# 1. KATEGORI
kategori_list = df['product_category'].unique()
kategori_input = pilih_opsi("Kategori", kategori_list)

# 2. TYPE
df_type = df[df['product_category'] == kategori_input]
type_list = df_type['product_type'].unique()
type_input = pilih_opsi("Type", type_list)

# 3. KOTA
city_list = df['city_location'].unique()
city_input = pilih_opsi("Kota", city_list)

# 4. KECAMATAN (filter dari kota)
df_sub = df[df['city_location'] == city_input]
subdistrict_list = df_sub['subdistrict_name'].unique()
subdistrict_input = pilih_opsi("Kecamatan", subdistrict_list)

# ENCODE INPUT
kategori_enc = le_category.transform([kategori_input])[0]
type_enc = le_type.transform([type_input])[0]
city_enc = le_city.transform([city_input])[0]
subdistrict_enc = le_subdistrict.transform([subdistrict_input])[0]

# FILTER DATA (REFERENSI HARGA)
subset = df[
    (df['product_category'] == kategori_input) &
    (df['product_type'] == type_input) &
    (df['city_location'] == city_input)
]

# fallback
if len(subset) < 5:
    subset = df[
        (df['product_category'] == kategori_input) &
        (df['product_type'] == type_input)
    ]

if len(subset) < 5:
    subset = df[df['product_category'] == kategori_input]

if len(subset) < 5:
    subset = df

# GENERATE HARGA
harga_mean = subset['unit_price'].mean()

harga_list = [
    harga_mean * 0.85,
    harga_mean,
    harga_mean * 1.15
]

# PREDIKSI
hasil = []

for harga in harga_list:
    data_input = pd.DataFrame([[ 
        city_enc,
        subdistrict_enc,
        kategori_enc,
        type_enc,
        harga,
        12,
        6,
        2
    ]], columns=X.columns)
    
    qty_pred = model.predict(data_input)[0]
    
    qty_bulan = qty_pred * 30
    omzet = harga * qty_bulan
    
    hasil.append((harga, qty_bulan, omzet))

# SORT
hasil = sorted(hasil, key=lambda x: x[2], reverse=True)

# OUTPUT
label = ["🔥 Paling Menguntungkan", "⚖️ Seimbang", "💎 Premium"]

print("\n=== REKOMENDASI HARGA ===\n")

for i, (harga, qty, omzet) in enumerate(hasil):
    print(label[i])
    print(f"Harga   : Rp {int(harga * 1000):,}")
    print(f"Terjual : {int(qty)} cup/bulan")
    print(f"Omset   : Rp {int(omzet * 1000):,}")
    print("-" * 30)
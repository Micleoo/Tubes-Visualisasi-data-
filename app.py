import streamlit as st
import pandas as pd
import numpy as np
import joblib

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.cluster import KMeans
from bokeh.palettes import Category10

# --- Load dataset ---
st.title("⚡ Analisis & Prediksi Konsumsi Energi Bangunan")

try:
    df = pd.read_csv("Energy_consumption_dataset.csv")
except Exception as e:
    st.error("❌ Gagal memuat data. Pastikan file `Energy_consumption_dataset.csv` tersedia.")
    st.stop()

# --- Validasi kolom penting ---
expected_columns = ['Temperature', 'Humidity', 'Occupancy', 'HVACUsage', 'LightingUsage', 'RenewableEnergy', 'EnergyConsumption', 'Month', 'Hour']
missing = [col for col in expected_columns if col not in df.columns]
if missing:
    st.error(f"❌ Dataset tidak memiliki kolom berikut: {', '.join(missing)}")
    st.stop()

# --- Bersih-bersih data ---
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['HVACUsage'] = df['HVACUsage'].map({'Off': 0, 'On': 1})
df['LightingUsage'] = df['LightingUsage'].map({'Off': 0, 'On': 1})

# --- Load model ---
try:
    model_regresi = joblib.load("model_regresi.pkl")
    model_clf = joblib.load("model_klasifikasi.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("❌ Gagal memuat salah satu model atau encoder. Pastikan semua file `.pkl` tersedia.")
    st.stop()

# --- Sidebar Navigasi ---
st.sidebar.title("🔍 Menu Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["Prediksi", "Klasifikasi", "Clustering", "Visualisasi"])

# --- Form input umum ---
def input_form():
    st.subheader("🧾 Input Parameter")
    temperature = st.slider("Temperature (°C)", 10, 40, 25)
    humidity = st.slider("Humidity (%)", 10, 100, 50)
    occupancy = st.slider("Occupancy (jumlah orang)", 0, 50, 10)
    hvac = st.selectbox("HVAC Usage", ["Off", "On"])
    lighting = st.selectbox("Lighting Usage", ["Off", "On"])
    renewable = st.slider("Renewable Energy (kWh)", 0, 100, 20)

    fitur = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Occupancy": occupancy,
        "HVACUsage": 1 if hvac == "On" else 0,
        "LightingUsage": 1 if lighting == "On" else 0,
        "RenewableEnergy": renewable
    }])
    return fitur

# --- Halaman Prediksi Regresi ---
if page == "Prediksi":
    st.header("🔋 Prediksi Konsumsi Energi")
    fitur = input_form()
    if st.button("Prediksi"):
        hasil = model_regresi.predict(fitur)[0]
        st.success(f"Perkiraan konsumsi energi: **{hasil:.2f} kWh**")

# --- Halaman Klasifikasi ---
elif page == "Klasifikasi":
    st.header("⚡ Klasifikasi Tingkat Konsumsi Energi")
    fitur = input_form()
    if st.button("Klasifikasikan"):
        kelas = model_clf.predict(fitur)[0]
        label = label_encoder.inverse_transform([kelas])[0]
        st.success(f"Konsumsi diprediksi berada di kelas: **{label}**")

# --- Halaman Clustering ---
elif page == "Clustering":
    st.header("🧠 Clustering Konsumsi Energi")

    fitur_scaled = scaler.transform(df[['Temperature', 'Humidity', 'Occupancy', 'HVACUsage', 'LightingUsage', 'RenewableEnergy']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(fitur_scaled)

    colors = [Category10[3][i] for i in df['Cluster']]
    source = ColumnDataSource(data=dict(
        x=df['Temperature'],
        y=df['EnergyConsumption'],
        cluster=[str(i) for i in df['Cluster']],
        color=colors
    ))

    try:
        p = figure(title="Clustering Konsumsi Energi", x_axis_label="Temperature", y_axis_label="Energy Consumption (kWh)", height=400, width=700)
        p.circle('x', 'y', color='color', legend_field='cluster', source=source, size=8)
        st.bokeh_chart(p)
    except Exception as e:
        st.error(f"❌ Gagal menampilkan grafik clustering: {e}")

# --- Halaman Visualisasi ---
elif page == "Visualisasi":
    st.header("📈 Visualisasi Interaktif Konsumsi Energi")

    # Rata-rata konsumsi per jam
    st.subheader("🔄 Konsumsi Energi per Jam berdasarkan Bulan")
    selected_month = st.selectbox("Pilih Bulan", sorted(df['Month'].unique()))
    filtered_df = df[df['Month'] == selected_month]
    hourly_avg = filtered_df.groupby("Hour")["EnergyConsumption"].mean().reset_index()
    source2 = ColumnDataSource(hourly_avg)

    p2 = figure(title=f"Konsumsi Energi per Jam - Bulan {selected_month}",
                x_axis_label="Jam", y_axis_label="Energy Consumption (kWh)",
                height=300, width=700)
    p2.line(x='Hour', y='EnergyConsumption', source=source2, line_width=2)
    p2.circle(x='Hour', y='EnergyConsumption', source=source2, size=8)
    p2.add_tools(HoverTool(tooltips=[("Jam", "@Hour"), ("Rata-rata", "@EnergyConsumption{0.00}")]))
    st.bokeh_chart(p2)

    # Rata-rata konsumsi per bulan
    st.subheader("📅 Rata-rata Konsumsi Energi per Bulan")
    monthly_avg = df.groupby("Month")["EnergyConsumption"].mean().reset_index()
    source = ColumnDataSource(monthly_avg)

    p1 = figure(title="Rata-rata Konsumsi Energi Bulanan",
                x_axis_label="Bulan", y_axis_label="Energy Consumption",
                x_range=[str(m) for m in sorted(df['Month'].unique())],
                height=300, width=700)
    p1.vbar(x='Month', top='EnergyConsumption', source=source, width=0.5)
    p1.add_tools(HoverTool(tooltips=[("Bulan", "@Month"), ("Rata-rata", "@EnergyConsumption{0.00}")]))
    st.bokeh_chart(p1)

    # Scatter Plot Dinamis
    st.subheader("📊 Scatter Plot Dinamis")
    numeric_cols = ['Temperature', 'Humidity', 'Occupancy', 'RenewableEnergy', 'EnergyConsumption']
    x_axis = st.selectbox("Pilih variabel X", numeric_cols, index=0)
    y_axis = st.selectbox("Pilih variabel Y", numeric_cols, index=4)

    scatter_source = ColumnDataSource(df)
    p3 = figure(title=f"{y_axis} vs {x_axis}",
                x_axis_label=x_axis, y_axis_label=y_axis,
                height=350, width=700)
    p3.circle(x=x_axis, y=y_axis, source=scatter_source, size=7, alpha=0.6, color="navy")
    p3.add_tools(HoverTool(tooltips=[(x_axis, f"@{x_axis}"), (y_axis, f"@{y_axis}")]))
    st.bokeh_chart(p3)

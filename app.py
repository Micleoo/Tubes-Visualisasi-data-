import streamlit as st
import pandas as pd
import numpy as np
import joblib
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.cluster import KMeans
from bokeh.palettes import Category10

st.set_page_config(page_title="Energy Analysis", layout="wide")

st.title("Analisis & Prediksi Konsumsi Energi Bangunan")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Energy_consumption_dataset.csv")
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df['HVACUsage'] = df['HVACUsage'].map({'Off': 0, 'On': 1})
        df['LightingUsage'] = df['LightingUsage'].map({'Off': 0, 'On': 1})
        return df
    except Exception as e:
        st.error("Gagal memuat data")
        st.stop()

@st.cache_resource
def load_models():
    try:
        return {
            'regresi': joblib.load("model_regresi.pkl"),
            'klasifikasi': joblib.load("model_klasifikasi.pkl"),
            'encoder': joblib.load("label_encoder.pkl"),
            'scaler': joblib.load("scaler.pkl")
        }
    except Exception as e:
        st.error("Gagal memuat model")
        st.stop()

df = load_data()
models = load_models()

with st.sidebar:
    st.header("🔍 Input Parameter")
    temp = st.slider("Temperature (°C)", 10, 40, 25)
    humidity = st.slider("Humidity (%)", 10, 100, 50)
    occupancy = st.slider("Occupancy", 0, 50, 10)
    hvac = st.selectbox("HVAC", ["Off", "On"])
    lighting = st.selectbox("Lighting", ["Off", "On"])
    renewable = st.slider("Renewable Energy", 0, 100, 20)
    
    fitur = pd.DataFrame([{
        "Temperature": temp, "Humidity": humidity, "Occupancy": occupancy,
        "HVACUsage": 1 if hvac == "On" else 0,
        "LightingUsage": 1 if lighting == "On" else 0,
        "RenewableEnergy": renewable
    }])

tab1, tab2, tab3, tab4 = st.tabs(["Prediksi", "Klasifikasi", "Clustering", "Visualisasi"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Prediksi Konsumsi", type="primary"):
            hasil = models['regresi'].predict(fitur)[0]
            st.success(f"**{hasil:.2f} kWh**")
       

with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Klasifikasi Tingkat", type="primary"):
            kelas = models['klasifikasi'].predict(fitur)[0]
            label = models['encoder'].inverse_transform([kelas])[0]
            st.success(f"**{label}**")

with tab3:
    fitur_cols = ['Temperature', 'Humidity', 'Occupancy', 'HVACUsage', 'LightingUsage', 'RenewableEnergy']
    fitur_scaled = models['scaler'].transform(df[fitur_cols])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(fitur_scaled)
    
    colors = [Category10[3][i] for i in df['Cluster']]
    source = ColumnDataSource(data=dict(
        x=df['Temperature'], y=df['EnergyConsumption'],
        cluster=[str(i) for i in df['Cluster']], color=colors
    ))
    
    p = figure(title="Clustering Konsumsi Energi", 
               x_axis_label="Temperature", y_axis_label="Energy Consumption",
               height=400, width=800)
    p.circle('x', 'y', color='color', legend_field='cluster', source=source, size=8)
    st.bokeh_chart(p, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Konsumsi Bulanan")
        monthly_avg = df.groupby("Month")["EnergyConsumption"].mean().reset_index()
        source1 = ColumnDataSource(monthly_avg)
        
        p1 = figure(title="Rata-rata Konsumsi per Bulan",
                    x_axis_label="Bulan", y_axis_label="Energy Consumption",
                    x_range=[str(m) for m in sorted(df['Month'].unique())],
                    height=300, width=400)
        p1.vbar(x='Month', top='EnergyConsumption', source=source1, width=0.5)
        p1.add_tools(HoverTool(tooltips=[("Bulan", "@Month"), ("Rata-rata", "@EnergyConsumption{0.00}")]))
        st.bokeh_chart(p1, use_container_width=True)

    with col2:
        st.subheader("Konsumsi per Jam")
        selected_month = st.selectbox("Pilih Bulan", sorted(df['Month'].unique()))
        filtered_df = df[df['Month'] == selected_month]
        hourly_avg = filtered_df.groupby("Hour")["EnergyConsumption"].mean().reset_index()
        source2 = ColumnDataSource(hourly_avg)
        
        p2 = figure(title=f"Konsumsi per Jam - Bulan {selected_month}",
                    x_axis_label="Jam", y_axis_label="Energy Consumption",
                    height=300, width=400)
        p2.line(x='Hour', y='EnergyConsumption', source=source2, line_width=2)
        p2.circle(x='Hour', y='EnergyConsumption', source=source2, size=6)
        p2.add_tools(HoverTool(tooltips=[("Jam", "@Hour"), ("Rata-rata", "@EnergyConsumption{0.00}")]))
        st.bokeh_chart(p2, use_container_width=True)
    
    st.subheader("Konsumsi Harian")
    if 'DayOfWeek' in df.columns:
        col_daily1, col_daily2 = st.columns([1, 2])
        with col_daily1:
            selected_month_daily = st.selectbox("Pilih Bulan untuk Analisis Harian", 
                                               sorted(df['Month'].unique()), 
                                               key="daily_month")
        with col_daily2:
            filtered_daily_df = df[df['Month'] == selected_month_daily]
            daily_avg = filtered_daily_df.groupby("DayOfWeek")["EnergyConsumption"].mean().reset_index()
            source_daily = ColumnDataSource(daily_avg)
            
            p_daily = figure(title=f"Konsumsi Harian - Bulan {selected_month_daily}",
                           x_axis_label="Hari", y_axis_label="Energy Consumption",
                           height=300, width=600)
            p_daily.vbar(x='DayOfWeek', top='EnergyConsumption', source=source_daily, width=0.6, color="green", alpha=0.7)
            p_daily.add_tools(HoverTool(tooltips=[("Hari", "@DayOfWeek"), ("Rata-rata", "@EnergyConsumption{0.00}")]))
            st.bokeh_chart(p_daily, use_container_width=True)
    else:
        st.info("Kolom 'DayOfWeek' tidak tersedia dalam dataset untuk visualisasi harian")
    
    st.subheader("Scatter Plot Dinamis")
    col3, col4, col5 = st.columns([1, 1, 3])
    
    with col3:
        numeric_cols = ['Temperature', 'Humidity', 'Occupancy', 'RenewableEnergy', 'EnergyConsumption']
        x_axis = st.selectbox("Variabel X", numeric_cols, index=0)
    with col4:
        y_axis = st.selectbox("Variabel Y", numeric_cols, index=4)
    with col5:
        scatter_source = ColumnDataSource(df)
        p3 = figure(title=f"{y_axis} vs {x_axis}",
                    x_axis_label=x_axis, y_axis_label=y_axis,
                    height=350, width=600)
        p3.circle(x=x_axis, y=y_axis, source=scatter_source, size=6, alpha=0.6, color="navy")
        p3.add_tools(HoverTool(tooltips=[(x_axis, f"@{x_axis}"), (y_axis, f"@{y_axis}")]))
        st.bokeh_chart(p3, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="Prediksi Jumlah Kendaraan",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3C72;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E86C1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #2E86C1;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>Sistem Prediksi Jumlah Kendaraan</div>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_traffic_model():
    return load_model('simpanmodel.h5')

# Cache the scaler function
@st.cache_resource
def get_scaler():
    # Load a sample of your data to fit the scaler
    df = pd.read_csv('traffic.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Hour'] = df['DateTime'].dt.hour
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    
    X = df[['Junction', 'Hour', 'Day', 'Month', 'DayOfWeek']]
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

try:
    model = load_traffic_model()
    scaler = get_scaler()
    st.sidebar.success("Model berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# Sidebar for inputs
st.sidebar.markdown("<div class='sub-header'>Parameter Input</div>", unsafe_allow_html=True)

# Junction selection
junction = st.sidebar.selectbox(
    "Pilih Junction / Persimpagan",
    options=[1, 2, 3, 4],
    help="Pilih persimpangan jalan"
)

# Date and time selection
date = st.sidebar.date_input(
    "Pilih Tanggal",
    value=datetime.now(),
    help="Pilih tanggal untuk prediksi"
)

hour = st.sidebar.slider(
    "Jam",
    min_value=0,
    max_value=23,
    value=datetime.now().hour,
    format="%d",
    help="Pilih jam untuk prediksi (0-23)"
)

# Extract features
selected_datetime = datetime.combine(date, datetime.min.time()) + timedelta(hours=hour)
day = selected_datetime.day
month = selected_datetime.month
day_of_week = selected_datetime.weekday()

# Button to make prediction
predict_button = st.sidebar.button("Prediksi", use_container_width=True)

# Information about the model
with st.sidebar.expander("Informasi Model"):
    st.markdown("""
    Model ini adalah Artificial Neural Network (ANN) untuk regresi yang memprediksi jumlah kendaraan berdasarkan:
    - Junction (Persimpangan)
    - Jam
    - Hari
    - Bulan
    - Hari dalam seminggu
    
    Model ini dilatih dengan data historis lalu lintas.
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='sub-header'>Visualisasi Data</div>", unsafe_allow_html=True)
    
    # Load some historical data for visualization
    @st.cache_data
    def load_historical_data():
        df = pd.read_csv('traffic.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    
    try:
        historical_data = load_historical_data()
        
        # Visualization options
        viz_option = st.selectbox(
            "Pilih Visualisasi Data ",
            ["Distribusi Jumlah Kendaraan per Junction", "Pola Lalu Lintas Harian", "Pola Lalu Lintas Mingguan"]
        )
        
        if viz_option == "Distribusi Jumlah Kendaraan per Junction":
            fig = px.box(historical_data, x="Junction", y="Vehicles", 
                        title="Distribusi Jumlah Kendaraan per Junction",
                        color="Junction", 
                        labels={"Junction": "Persimpangan", "Vehicles": "Jumlah Kendaraan"})
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Pola Lalu Lintas Harian":
            hourly_data = historical_data.copy()
            hourly_data['Hour'] = hourly_data['DateTime'].dt.hour
            hourly_avg = hourly_data.groupby(['Junction', 'Hour'])['Vehicles'].mean().reset_index()
            
            fig = px.line(hourly_avg, x="Hour", y="Vehicles", color="Junction",
                        title="Rata-rata Jumlah Kendaraan per Jam",
                        labels={"Hour": "Jam", "Vehicles": "Rata-rata Jumlah Kendaraan", "Junction": "Persimpangan"})
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_option == "Pola Lalu Lintas Mingguan":
            weekly_data = historical_data.copy()
            weekly_data['DayOfWeek'] = weekly_data['DateTime'].dt.dayofweek
            days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
            weekly_avg = weekly_data.groupby(['Junction', 'DayOfWeek'])['Vehicles'].mean().reset_index()
            
            fig = px.line(weekly_avg, x="DayOfWeek", y="Vehicles", color="Junction",
                        title="Rata-rata Jumlah Kendaraan per Hari dalam Seminggu",
                        labels={"DayOfWeek": "Hari", "Vehicles": "Rata-rata Jumlah Kendaraan", "Junction": "Persimpangan"},
                        category_orders={"DayOfWeek": list(range(7))})
            
            # Update x-axis labels to day names
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(7)), ticktext=days))
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading historical data: {e}")

with col2:
    st.markdown("<div class='sub-header'>Hasil Prediksi</div>", unsafe_allow_html=True)
    
    # Display selected parameters
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("**Parameter yang dipilih:**")
    st.write(f"📍 **Junction:** {junction}")
    st.write(f"📅 **Tanggal:** {date.strftime('%d-%m-%Y')}")
    st.write(f"🕒 **Jam:** {hour}:00")
    st.write(f"📊 **Hari ke-{day}**")
    st.write(f"📆 **Bulan ke-{month}**")
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    st.write(f"📆 **Hari:** {days[day_of_week]}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Make prediction when button is clicked
    if predict_button:
        with st.spinner('Sedang memprediksi...'):
            # Prepare input for prediction
            input_data = np.array([[junction, hour, day, month, day_of_week]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0][0]
            
            # Show prediction result
            st.markdown("<div class='sub-header' style='margin-top: 2rem;'>Hasil:</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.5rem; font-weight: bold; color: #1E3C72; text-align: center;'>{prediction:.0f}</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; font-size: 1.2rem;'>kendaraan</div>", unsafe_allow_html=True)
            
            # Confidence gauge
            # This is a simple approximation - in a real app you might want to use model uncertainty
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = min(100, max(0, 100 - abs(prediction - 20) / 2)),  # Simple metric for demonstration
                title = {'text': "Tingkat Keyakinan Atau Akurasi", 'font': {'size': 24}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#2E86C1"},
                    'steps': [
                        {'range': [0, 30], 'color': '#FFCCCC'},
                        {'range': [30, 70], 'color': '#FFFFCC'},
                        {'range': [70, 100], 'color': '#CCFFCC'},
                    ],
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add some context to the prediction
            if prediction < 10:
                st.info("Lalu lintas diperkirakan lancar dengan kepadatan rendah.")
            elif prediction < 20:
                st.info("Lalu lintas diperkirakan normal.")
            elif prediction < 30:
                st.warning("Lalu lintas diperkirakan cukup padat.")
            else:
                st.error("Lalu lintas diperkirakan sangat padat. Pertimbangkan untuk mencari rute alternatif.")

# Footer
st.markdown("<div class='footer'>© 2025 Sistem Prediksi Lalu Lintas. Dibuat dengan Streamlit.</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import warnings
import random
import time

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === PAGE CONFIG ===
st.set_page_config(page_title="📊 Task Dashboard", layout="wide", page_icon="📌")

# === SESSION STATE BASED REFRESH ===
if "refresh_trigger" not in st.session_state:
    st.session_state.refresh_trigger = 0

if st.button("🔄 Refresh Data"):
    st.session_state.refresh_trigger = random.randint(0, 100000)

# === FORCE RERUN BEFORE LOADING DATA ===
trigger = st.session_state.refresh_trigger

# === LOAD DATA ===
df = pd.read_csv("live_data.csv")
df.columns = df.columns.str.strip()
df.rename(columns={'Employee': 'Name', 'Type.1': 'Task'}, inplace=True)

if 'Task Lat/Lng' in df.columns:
    lat_lng_split = df['Task Lat/Lng'].str.split(",", expand=True)
    df['Latitude'] = pd.to_numeric(lat_lng_split[0], errors='coerce')
    df['Longitude'] = pd.to_numeric(lat_lng_split[1], errors='coerce')

# === FILTERS ===
st.sidebar.header("📁 Filters")
filters = ['Branch', 'CLUSTER', 'TERRITORY NAME', 'Task', 'Status']
for col in filters:
    if col in df.columns:
        selection = st.sidebar.multiselect(f"Filter by {col}", options=df[col].dropna().unique())
        if selection:
            df = df[df[col].isin(selection)]

df['Status'] = df['Status'].astype(str).str.upper()

# === KPIs ===
st.title("📊 Task Status Dashboard")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Tasks", len(df))
k2.metric("✅ Completed", df['Status'].eq('COMPLETED').sum())
k3.metric("🕒 In Progress", df['Status'].eq('IN PROGRESS').sum())
k4.metric("⏳ Pending", df['Status'].eq('PENDING').sum())
st.markdown("---")

# === PIE & BAR CHARTS ===
c1, c2 = st.columns(2)
with c1:
    pie_df = df['Status'].value_counts().reset_index()
    pie_df.columns = ['Status', 'Count']
    fig = px.pie(pie_df, values='Count', names='Status', hole=0.3)
    fig.update_layout(title="Task Status Distribution")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    if 'Task' in df.columns:
        bar = df.groupby(['Task', 'Status']).size().reset_index(name='Count')
        fig2 = px.bar(bar, x='Task', y='Count', color='Status', barmode='stack')
        st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

# === DATE TREND ===
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    trend = df.groupby([df['Date'].dt.date, 'Status']).size().reset_index(name='Count')
    fig3 = px.line(trend, x='Date', y='Count', color='Status', title="📈 Task Trend Over Time")
    st.plotly_chart(fig3, use_container_width=True)

# === TERRITORY BAR ===
if 'TERRITORY NAME' in df.columns:
    terr = df.groupby(['TERRITORY NAME', 'Status']).size().reset_index(name='Count')
    fig4 = px.bar(terr, x='TERRITORY NAME', y='Count', color='Status', barmode='group')
    fig4.update_layout(xaxis_tickangle=-45, title="📌 Task Status by Territory")
    st.plotly_chart(fig4, use_container_width=True)

# === INTERACTIVE MAP WITH GEOJSON ===
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    st.subheader("🗺️ Interactive Task Map with Boundary Overlays")
    map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    center_lat = map_df['Latitude'].mean() if not map_df.empty else 10.85
    center_lon = map_df['Longitude'].mean() if not map_df.empty else 76.27
    fig_map = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude",
                                color="Status", hover_name="Name",
                                hover_data={"Task": True, "Status": True},
                                zoom=6, height=600)
    fig_map.update_layout(mapbox_style="open-street-map",
                          mapbox_center={"lat": center_lat, "lon": center_lon},
                          margin=dict(l=0, r=0, t=0, b=0))
    try:
        with open("boundaries.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        fig_map.update_layout(mapbox_layers=[{
            "source": geojson_data,
            "type": "line",
            "color": "blue",
            "opacity": 0.5,
            "line": {"width": 2}
        }])
    except Exception as e:
        st.warning(f"🔹 Note: GeoJSON overlay not applied: {e}")
    st.plotly_chart(fig_map, use_container_width=True)

# === FRAUD DETECTION MAP ===
st.markdown("---")
st.subheader("🧭 Suspected Task Same Locations")
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    coord_counts = df.groupby(['Latitude', 'Longitude']).size().reset_index(name='Count')
    fraud_coords = coord_counts[coord_counts['Count'] > 1]
    if not fraud_coords.empty:
        fraud_df = df.merge(fraud_coords, on=['Latitude', 'Longitude'], how='inner')
        st.info(f"🚨 {len(fraud_df)} records found with duplicate task locations.")
        fig_fraud = px.scatter_mapbox(fraud_df, lat="Latitude", lon="Longitude",
                                      color="Status", hover_name="Name",
                                      hover_data={"Task": True, "Status": True, "Date": True},
                                      zoom=6, height=500)
        fig_fraud.update_layout(mapbox_style="open-street-map",
                                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                title="⚠️ Same-Location Task Incidents")
        st.plotly_chart(fig_fraud, use_container_width=True)
        st.markdown("#### 🔍 Suspected Task Table")
        show_cols = ['Name', 'Task', 'Status', 'Latitude', 'Longitude', 'Date', 'TERRITORY NAME', 'CLUSTER', 'Branch']
        show_cols = [col for col in show_cols if col in fraud_df.columns]
        st.dataframe(fraud_df[show_cols].sort_values(['Latitude', 'Longitude']), use_container_width=True)
    else:
        st.success("✅ No duplicate-location tasks found.")

# === DOWNLOAD ===
st.sidebar.markdown("---")
st.sidebar.download_button("📥 Download Filtered Data", data=df.to_csv(index=False), file_name="filtered_data.csv")

# === ML PLACEHOLDER ===
st.markdown("---")
st.markdown("### 🤖 Machine Learning Insights")
st.info("ML models not yet implemented. Add prediction hooks here.")

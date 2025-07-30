import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import DBSCAN
import warnings
import uuid
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_page_config(page_title="📊 Task Analytics Dashboard", layout="wide", page_icon="📈")

# Custom CSS for Tailwind-like styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        font-size: 1.875rem;
        font-weight: 700;
        color: #2563eb;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
# Manual cache clear
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

# Load Data Function
@st.cache_data(ttl=3600)  # Refresh every hour
def load_data():
    df = pd.read_csv("live_data.csv")
    df.columns = df.columns.str.strip()

    col_map = {
        'CLUSTER': 'CLUSTER', 'Branch': 'Branch', 'Territory_Name': 'Territory_Name',
        'Task_Type': 'Task_Type', 'Task_Status': 'Task_Status', 'Branch_ID': 'Branch_ID'
    }
    for newcol, oldcol in col_map.items():
        if oldcol not in df.columns:
            df[newcol] = None

    if 'Task_Lat_Lng' in df.columns:
        latlng_split = df['Task_Lat_Lng'].astype(str).str.split(",", expand=True)
        df['Latitude'] = pd.to_numeric(latlng_split[0], errors='coerce')
        df['Longitude'] = pd.to_numeric(latlng_split[1], errors='coerce')

    for dt_col in ['Task_Start', 'Created_Date', 'ETL_Run_Date']:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], format='%d-%m-%Y %H:%M', errors='coerce').fillna(
                pd.to_datetime(df[dt_col], format='%d-%m-%Y', errors='coerce')
            )

    if 'Time_Taken' in df.columns:
        df['Time_Taken_Minutes'] = pd.to_timedelta(df['Time_Taken'], errors='coerce').dt.total_seconds() / 60

    if 'Task_Frequency' in df.columns:
        df['Task_Frequency_Minutes'] = pd.to_timedelta(df['Task_Frequency'], errors='coerce').dt.total_seconds() / 60

    df['Task_Status'] = df['Task_Status'].astype(str).str.upper()
    df['Delayed'] = df['Delayed'].astype(str).str.title()
    df['Follow_Up_Status'] = df['Follow_Up_Status'].astype(str).str.title()

    return df

# Call the function after it's defined
df = load_data()


# --- Sidebar Filters ---
with st.sidebar:
    st.header("📁 Filters", anchor=False)
    territory_options = ["All"] + sorted(df['Territory_Name'].dropna().unique().tolist())
    territory = st.selectbox("Territory", options=territory_options, key="territory")
    filtered_df = df.copy()
    if territory != "All":
        filtered_df = filtered_df[filtered_df['Territory_Name'] == territory]
    cluster_options = ["All"] + sorted(filtered_df['CLUSTER'].dropna().unique().tolist())
    cluster = st.selectbox("Cluster", options=cluster_options, key="cluster")
    if cluster != "All":
        filtered_df = filtered_df[filtered_df['CLUSTER'] == cluster]
    branch_options = ["All"] + sorted(filtered_df['Branch'].dropna().unique().tolist())
    branch = st.selectbox("Branch", options=branch_options, key="branch")
    if branch != "All":
        filtered_df = filtered_df[filtered_df['Branch'] == branch]
    task_options = ["All"] + sorted(filtered_df['Task_Type'].dropna().unique().tolist())
    task = st.selectbox("Task Type", options=task_options, key="task_type")
    if task != "All":
        filtered_df = filtered_df[filtered_df['Task_Type'] == task]
    status_options = ["All"] + sorted(filtered_df['Task_Status'].dropna().unique().tolist())
    status = st.selectbox("Status", options=status_options, key="status")
    if status != "All":
        filtered_df = filtered_df[filtered_df['Task_Status'] == status]
    designation_options = ["All"] + sorted(filtered_df['Designation'].dropna().unique().tolist())
    designation = st.selectbox("Designation", options=designation_options, key="designation")
    if designation != "All":
        filtered_df = filtered_df[filtered_df['Designation'] == designation]
    zone_options = ["All"] + sorted(filtered_df['Zone'].dropna().unique().tolist())
    zone = st.selectbox("Zone", options=zone_options, key="zone")
    if zone != "All":
        filtered_df = filtered_df[filtered_df['Zone'] == zone]
    delayed_options = ["All"] + sorted(filtered_df['Delayed'].dropna().unique().tolist())
    delayed = st.selectbox("Delayed", options=delayed_options, key="delayed")
    if delayed != "All":
        filtered_df = filtered_df[filtered_df['Delayed'] == delayed]
    follow_up_options = ["All"] + sorted(filtered_df['Follow_Up_Status'].dropna().unique().tolist())
    follow_up = st.selectbox("Follow Up Status", options=follow_up_options, key="follow_up")
    if follow_up != "All":
        filtered_df = filtered_df[filtered_df['Follow_Up_Status'] == follow_up]
    df = filtered_df.copy()

# --- KPI Metrics ---
st.markdown("<div class='section-header'>📊 Task Analytics Dashboard</div>", unsafe_allow_html=True)
total = len(df)
comp = df['Task_Status'].eq('COMPLETED').sum()
pend = df['Task_Status'].eq('PENDING').sum()
prog = df['Task_Status'].eq('IN PROGRESS').sum()
delayed_count = df['Delayed'].eq('Yes').sum()
avg_accuracy = df['Accuracy_Percentage'].mean().round(1) if 'Accuracy_Percentage' in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>Total Tasks</h3>
        <p>{total}</p>
    </div>
    """, unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>✅ Completed</h3>
        <p>{comp}</p>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>🕒 In Progress</h3>
        <p>{prog}</p>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>⏳ Pending</h3>
        <p>{pend}</p>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class='metric-card'>
        <h3>🚨 Delayed</h3>
        <p>{delayed_count}</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

st.markdown("<div class='section-header'>📌 Territory × Task Type Matrix</div>", unsafe_allow_html=True)
if 'Territory_Name' in df.columns and 'Task_Type' in df.columns:
    terr_type_pivot = pd.pivot_table(
        df,
        index='Territory_Name',
        columns='Task_Type',
        values='Task_ID' if 'Task_ID' in df.columns else df.columns[0],
        aggfunc='count',
        fill_value=0,
        margins=False
    )
    # --- Add a Total Row (sum of each column) ---
    total_row = pd.DataFrame(terr_type_pivot.sum(axis=0)).T
    total_row.index = ['Total']
    terr_type_pivot_total = pd.concat([terr_type_pivot, total_row])

    st.dataframe(terr_type_pivot_total, use_container_width=True)
    st.download_button(
        "Download Territory × Type Table (with Totals) (CSV)",
        terr_type_pivot_total.to_csv(),
        file_name="territory_type_matrix_with_total.csv",
        mime="text/csv",
        key=f"download_terrtype_{uuid.uuid4()}"
    )
else:
    st.info("Required columns for Territory × Type Matrix not found.")
st.markdown("---")


# --- TERRITORY, CLUSTER, TYPE, GRAND TOTAL & % COMPLETED TABLE ---
st.markdown("<div class='section-header'>🧮 Performance by Territory, Cluster & Status</div>", unsafe_allow_html=True)
if set(['Territory_Name', 'CLUSTER', 'Task_Status']).issubset(df.columns):
    # Pivot: rows are Territory, Cluster; columns are Task_Status; values = count
    perf_pivot = pd.pivot_table(
        df,
        index=['Territory_Name', 'CLUSTER'],
        columns='Task_Status',
        values='Task_ID' if 'Task_ID' in df.columns else df.columns[0],
        aggfunc='count',
        fill_value=0,
        margins=False
    ).reset_index()
    
    # Ensure key statuses are always shown (fill with 0 if missing)
    for status in ['COMPLETED', 'IN PROGRESS', 'PENDING']:
        if status not in perf_pivot.columns:
            perf_pivot[status] = 0

    # Calculate Grand Total per row
    status_cols = [col for col in perf_pivot.columns if col not in ['Territory_Name', 'CLUSTER']]
    perf_pivot['GRAND TOTAL'] = perf_pivot[status_cols].sum(axis=1)

    # Calculate % Completed
    perf_pivot['% COMPLETED'] = (
        perf_pivot['COMPLETED'] / perf_pivot['GRAND TOTAL'] * 100
    ).round(1).fillna(0)

    # Reorder columns for clarity
    col_order = (
        ['Territory_Name', 'CLUSTER'] + 
        [c for c in ['COMPLETED', 'IN PROGRESS', 'PENDING'] if c in perf_pivot.columns] +
        [c for c in status_cols if c not in ['COMPLETED', 'IN PROGRESS', 'PENDING']] +
        ['GRAND TOTAL', '% COMPLETED']
    )
    perf_pivot = perf_pivot[col_order]

    st.dataframe(perf_pivot, use_container_width=True)
    st.download_button(
        "Download Territory/Cluster/Status Breakdown (CSV)",
        perf_pivot.to_csv(index=False),
        file_name="territory_cluster_status_summary.csv",
        mime="text/csv",
        key=f"download_tcstatus_{uuid.uuid4()}"
    )
else:
    st.info("Required columns for status performance breakdown not found.")
st.markdown("---")


# Add this above the new table
territory_values = ['All'] + sorted(df['Territory_Name'].dropna().unique().tolist())
selected_territory = st.selectbox("Filter by Territory (for Cluster-Branch Table)", territory_values, key="clusterbranch_territory")

if selected_territory != 'All':
    filtered_df_cb = df[df['Territory_Name'] == selected_territory]
else:
    filtered_df_cb = df.copy()

st.markdown("<div class='section-header'>🗂️ Cluster, Branch, Task Status Performance</div>", unsafe_allow_html=True)
if set(['CLUSTER', 'Branch', 'Task_Status']).issubset(filtered_df_cb.columns):
    cb_pivot = pd.pivot_table(
        filtered_df_cb,
        index=['CLUSTER', 'Branch'],
        columns='Task_Status',
        values='Task_ID' if 'Task_ID' in filtered_df_cb.columns else filtered_df_cb.columns[0],
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Ensure all main status columns present
    for status in ['COMPLETED', 'IN PROGRESS', 'PENDING']:
        if status not in cb_pivot.columns:
            cb_pivot[status] = 0

    # Total & % Completed
    status_cols = [c for c in cb_pivot.columns if c not in ['CLUSTER', 'Branch']]
    cb_pivot['TOTAL'] = cb_pivot[status_cols].sum(axis=1)
    cb_pivot['% COMPLETED'] = (
        cb_pivot['COMPLETED'] / cb_pivot['TOTAL'] * 100
    ).round(1).fillna(0)

    # Reorder columns: Cluster, Branch, (statuses), TOTAL, %
    col_order = (
        ['CLUSTER', 'Branch'] + 
        [c for c in ['COMPLETED', 'IN PROGRESS', 'PENDING'] if c in cb_pivot.columns] +
        [c for c in status_cols if c not in ['COMPLETED', 'IN PROGRESS', 'PENDING']] +
        ['TOTAL', '% COMPLETED']
    )
    cb_pivot = cb_pivot[col_order]

    st.dataframe(cb_pivot, use_container_width=True)
    st.download_button(
        "Download Cluster/Branch/Status Breakdown (CSV)",
        cb_pivot.to_csv(index=False),
        file_name="cluster_branch_status_summary.csv",
        mime="text/csv",
        key=f"download_cbstatus_{uuid.uuid4()}"
    )
else:
    st.info("Required columns for Cluster-Branch performance not found.")
st.markdown("---")



# --- Charts ---
c1, c2 = st.columns(2)
with c1:
    pie_df = pd.DataFrame({
        'Status': ['Completed', 'In Progress', 'Pending'],
        'Count': [comp, prog, pend]
    })
    fig = px.pie(
        pie_df, values='Count', names='Status', hole=0.3,
        color_discrete_sequence=['#1e3a8a', '#8b5cf6', '#e11d48']
    )
    fig.update_layout(
        title="Task Status Distribution",
        title_x=0.5,
        font=dict(size=14),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    if 'Task_Type' in df.columns and 'Task_Status' in df.columns:
        bar = df.groupby(['Task_Type', 'Task_Status']).size().reset_index(name='Count')
        fig2 = px.bar(
            bar, x='Task_Type', y='Count', color='Task_Status', barmode='stack',
            color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6']
        )
        fig2.update_layout(
            title="Tasks by Type and Status",
            title_x=0.5,
            xaxis_title="Task Type",
            yaxis_title="Count",
            xaxis_tickangle=-45,
            font=dict(size=14)
        )
        st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

# --- Accuracy Distribution ---
if 'Accuracy_Percentage' in df.columns:
    st.markdown("<div class='section-header'>📏 Accuracy Distribution</div>", unsafe_allow_html=True)
    fig_accuracy = px.histogram(
        df, x='Accuracy_Percentage', nbins=20, title="Distribution of Task Accuracy",
        color_discrete_sequence=['#2563eb']
    )
    fig_accuracy.update_layout(
        title_x=0.5,
        xaxis_title="Accuracy Percentage",
        yaxis_title="Count",
        font=dict(size=14)
    )
    st.plotly_chart(fig_accuracy, use_container_width=True)
    st.markdown("---")

# --- Delayed Tasks Analysis ---
if 'Delayed' in df.columns:
    st.markdown("<div class='section-header'>⏰ Delayed Tasks Analysis</div>", unsafe_allow_html=True)
    delayed_df = df.groupby(['Task_Type', 'Delayed']).size().reset_index(name='Count')
    fig_delayed = px.bar(
        delayed_df, x='Task_Type', y='Count', color='Delayed', barmode='group',
        color_discrete_sequence=['#ef4444', '#22c55e']
    )
    fig_delayed.update_layout(
        title="Delayed vs Non-Delayed Tasks by Type",
        title_x=0.5,
        xaxis_title="Task Type",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        font=dict(size=14)
    )
    st.plotly_chart(fig_delayed, use_container_width=True)
    st.markdown("---")

# --- DBSCAN for Employee Clusters ---
MIN_CLUSTERS = 3
CLUSTER_RADIUS_METERS = 500
coord_df = df.dropna(subset=['Latitude', 'Longitude', 'Employee']).copy()
if not coord_df.empty:
    coords_rad = np.radians(coord_df[['Latitude', 'Longitude']].values)
    db = DBSCAN(eps=CLUSTER_RADIUS_METERS/6371000, min_samples=1, algorithm='ball_tree', metric='haversine')
    clusters = db.fit_predict(coords_rad)
    coord_df['ClusterID'] = clusters
    emp_clusters = coord_df.groupby('Employee')['ClusterID'].nunique()
    valid_employees = set(emp_clusters[emp_clusters >= MIN_CLUSTERS].index)
else:
    valid_employees = set()

# --- DBSCAN for Map (Fraud Analysis) ---
st.markdown("<div class='section-header'>🗺️ Task Location Analysis</div>", unsafe_allow_html=True)
map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
only_suspected_employees = set()
if not map_df.empty:
    coords = map_df[['Latitude', 'Longitude']].values
    coords_rad = np.radians(coords)
    db_map = DBSCAN(eps=CLUSTER_RADIUS_METERS/6371000, min_samples=2, algorithm='ball_tree', metric='haversine')
    clusters_map = db_map.fit_predict(coords_rad)
    map_df['Cluster'] = clusters_map
    map_df['Suspected'] = map_df['Cluster'] != -1

    df = df.merge(map_df[['Latitude', 'Longitude', 'Suspected', 'Employee']],
                  on=['Latitude', 'Longitude', 'Employee'], how='left', suffixes=('', '_y'))

    coords_df = df.dropna(subset=['Latitude', 'Longitude'])
    for emp, group in coords_df.groupby('Employee'):
        if not group.empty and group['Suspected'].all():
            only_suspected_employees.add(emp)

    emp_with_coords = set(coords_df['Employee'])
    all_emps = set(df['Employee'].dropna())
    no_coords_emps = all_emps - emp_with_coords
    only_suspected_employees.update(no_coords_emps)

    center_lat = map_df['Latitude'].mean()
    center_lon = map_df['Longitude'].mean()
    map_df['Marker_Color'] = map_df.apply(
        lambda row: 'red' if row['Suspected'] else (
            'green' if row['Task_Status'] == 'COMPLETED' else
            'orange' if row['Task_Status'] == 'IN PROGRESS' else
            'blue' if row['Task_Status'] == 'PENDING' else 'gray'
        ), axis=1
    )
    fig_map = px.scatter_mapbox(
        map_df,
        lat="Latitude", lon="Longitude",
        color="Marker_Color",
        hover_name="Task_ID",
        hover_data={
            "Task_Type": True, "Task_Status": True, "Branch": True, "Territory_Name": True,
            "CLUSTER": True, "Mobile_Number": True, "Title": True, "Employee": True
        },
        zoom=6, height=600,
        color_discrete_map={"red": "#ef4444", "green": "#22c55e", "orange": "#f97316", "blue": "#3b82f6", "gray": "#9ca3af"}
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title="Task Locations (500m DBSCAN Clustering)",
        title_x=0.5,
        font=dict(size=14)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    fraud_df = map_df[map_df['Suspected']]
    if not fraud_df.empty:
        st.warning(f"🚨 {len(fraud_df)} records found with suspected same task locations (within 500m, marked **red** on map).")
        st.dataframe(fraud_df, use_container_width=True)
        st.download_button(
            label="Download Suspected Locations (CSV)",
            data=fraud_df.to_csv(index=False),
            file_name="suspected_locations.csv",
            mime="text/csv",
            key=f"download_suspected_{uuid.uuid4()}"
        )
    else:
        st.success("✅ No duplicate-location tasks found within 500m.")
else:
    st.success("✅ No GPS data to check for duplicate locations.")
st.markdown("---")

# --- Best & Poor Performer Analysis ---
st.markdown("<div class='section-header'>🏆 Performance Analysis</div>", unsafe_allow_html=True)
emp_cols = ['Employee', 'Mobile_Number', 'Branch', 'CLUSTER', 'Territory_Name', 'Task_Type', 'Task_Status']
emp_exists = all([col in df.columns for col in emp_cols])
if emp_exists:
    emp_task = df.groupby(['Employee', 'Mobile_Number', 'Branch', 'CLUSTER', 'Territory_Name', 'Designation']).agg(
        Total_Tasks=('Task_Status', 'count'),
        Completed=('Task_Status', lambda x: (x=='COMPLETED').sum()),
        Pending=('Task_Status', lambda x: (x=='PENDING').sum()),
        In_Progress=('Task_Status', lambda x: (x=='IN PROGRESS').sum()),
        Avg_Accuracy=('Accuracy_Percentage', 'mean'),
        Avg_Time_Taken=('Time_Taken_Minutes', 'mean'),
        Delayed_Tasks=('Delayed', lambda x: (x=='Yes').sum())
    ).reset_index()
    emp_task['Completion_%'] = (emp_task['Completed'] / emp_task['Total_Tasks'] * 100).round(1)
    emp_task['Avg_Accuracy'] = emp_task['Avg_Accuracy'].round(1)
    emp_task['Avg_Time_Taken'] = emp_task['Avg_Time_Taken'].round(1)
    emp_task_filt = emp_task[emp_task['Total_Tasks'] >= 5].copy()
    emp_task_filt = emp_task_filt[emp_task_filt['Employee'].isin(valid_employees)]
    best_emps = emp_task_filt.sort_values(['Completion_%', 'Avg_Accuracy', 'Completed'], ascending=[False, False, False]).head(5)
    poor_emps = emp_task_filt.sort_values(['Completion_%', 'Delayed_Tasks', 'Total_Tasks'], ascending=[True, False, False]).head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌟 Top 5 Best Employees")
        st.dataframe(best_emps, use_container_width=True)
        st.download_button(
            label="Download Best Employees (CSV)",
            data=best_emps.to_csv(index=False),
            file_name="best_employees.csv",
            mime="text/csv",
            key=f"download_best_{uuid.uuid4()}"
        )
    with c2:
        st.subheader("❗ Bottom 5 Poor Employees")
        st.dataframe(poor_emps, use_container_width=True)
        st.download_button(
            label="Download Poor Employees (CSV)",
            data=poor_emps.to_csv(index=False),
            file_name="poor_employees.csv",
            mime="text/csv",
            key=f"download_poor_{uuid.uuid4()}"
        )

    # --- Branch Level Analysis ---
    br_cols = ['Branch', 'CLUSTER', 'Territory_Name', 'Zone']
    branch_task = df.groupby(br_cols).agg(
        Total_Tasks=('Task_Status', 'count'),
        Completed=('Task_Status', lambda x: (x=='COMPLETED').sum()),
        Pending=('Task_Status', lambda x: (x=='PENDING').sum()),
        In_Progress=('Task_Status', lambda x: (x=='IN PROGRESS').sum()),
        Avg_Accuracy=('Accuracy_Percentage', 'mean'),
        Delayed_Tasks=('Delayed', lambda x: (x=='Yes').sum())
    ).reset_index()
    branch_task['Completion_%'] = (branch_task['Completed'] / branch_task['Total_Tasks'] * 100).round(1)
    branch_task['Avg_Accuracy'] = branch_task['Avg_Accuracy'].round(1)
    branch_task_filt = branch_task[branch_task['Total_Tasks'] >= 10].copy()
    best_branch = branch_task_filt.sort_values(['Completion_%', 'Avg_Accuracy', 'Completed'], ascending=[False, False, False]).head(3)
    poor_branch = branch_task_filt.sort_values(['Completion_%', 'Delayed_Tasks', 'Total_Tasks'], ascending=[True, False, False]).head(3)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🏢 Top 3 Best Branches")
        st.dataframe(best_branch, use_container_width=True)
    with c2:
        st.subheader("🚩 Bottom  fondly Branch")
        st.dataframe(poor_branch, use_container_width=True)
st.markdown("---")

# --- Cluster-wise Status Charts ---
if 'CLUSTER' in df.columns and 'Task_Status' in df.columns:
    st.markdown("<div class='section-header'>🏢 Cluster-wise Analysis</div>", unsafe_allow_html=True)
    cluster_status = df.groupby(['CLUSTER', 'Task_Status']).size().reset_index(name='Count')
    fig_cbar = px.bar(
        cluster_status, x='CLUSTER', y='Count', color='Task_Status',
        barmode='stack', title='Tasks by Status per Cluster',
        color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6']
    )
    fig_cbar.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        font=dict(size=14)
    )
    st.plotly_chart(fig_cbar, use_container_width=True)
    st.dataframe(
        cluster_status.pivot(index='CLUSTER', columns='Task_Status', values='Count').fillna(0),
        use_container_width=True
    )
st.markdown("---")

# --- Date Trend ---
if 'Task_Start' in df.columns:
    st.markdown("<div class='section-header'>📈 Task Trend Over Time</div>", unsafe_allow_html=True)
    df['Task_Date'] = pd.to_datetime(df['Task_Start'], errors='coerce').dt.date
    valid_trend = df.dropna(subset=['Task_Date', 'Task_Status'])
    if not valid_trend.empty:
        trend = (
            valid_trend.groupby(['Task_Date', 'Task_Status'])
            .size()
            .reset_index(name='Count')
            .sort_values('Task_Date')
        )
        if len(trend) >= 1:
            fig3 = px.line(
                trend,
                x='Task_Date',
                y='Count',
                color='Task_Status',
                title="Task Trend Over Time",
                color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6']
            )
            fig3.update_traces(mode='lines+markers')
            fig3.update_layout(
                title_x=0.5,
                font=dict(size=14),
                hovermode="x unified"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data available for trend chart.")
    else:
        st.info("No valid dates found in Task_Start for trend chart.")
else:
    st.warning("Task_Start column is missing from data.")
st.markdown("---")

# --- Custom Pivot & Chart Lab ---
st.markdown("<div class='section-header'>🧩 Custom Analysis Lab</div>", unsafe_allow_html=True)
with st.expander("Click to open Pivot Lab", expanded=False):
    cols = list(df.columns)
    if len(cols) < 3:
        st.info("Not enough columns for pivot analysis.")
    else:
        index_col = st.selectbox("Row (Index)", cols, key="pivot_row")
        column_col = st.selectbox("Column", cols, key="pivot_col")
        value_col = st.selectbox("Value", cols, key="pivot_val")
        agg_func = st.selectbox("Aggregation Function", ['count', 'sum', 'mean', 'min', 'max'], key="pivot_agg")
        try:
            pivot_table = pd.pivot_table(
                df,
                index=index_col,
                columns=column_col,
                values=value_col,
                aggfunc=agg_func,
                fill_value=0
            )
            st.dataframe(pivot_table, use_container_width=True)
            st.download_button(
                label="Download Pivot Table (CSV)",
                data=pivot_table.to_csv(),
                file_name="pivot_table.csv",
                mime="text/csv",
                key=f"download_pivot_{uuid.uuid4()}"
            )
            chart_type = st.selectbox("Chart Type", ['Bar', 'Heatmap', 'Pie'], key="pivot_chart")
            if chart_type == 'Bar':
                plot_df = pivot_table.reset_index().melt(id_vars=index_col, var_name=column_col, value_name='Value')
                fig = px.bar(
                    plot_df, x=index_col, y='Value', color=column_col, barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Heatmap':
                fig = px.imshow(
                    pivot_table,
                    labels=dict(x=column_col, y=index_col, color=value_col),
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Pie':
                pie_data = pivot_table.sum(axis=1).reset_index(name='Value')
                fig = px.pie(
                    pie_data, names=index_col, values='Value',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not create pivot: {e}")
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

# ======= Styles =======
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; font-family: 'Inter', sans-serif; }
    .sidebar .sidebar-content {
        background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 0.375rem;
        padding: 0.5rem 1rem; font-weight: 500; transition: background-color 0.2s;
    }
    .stButton>button:hover { background-color: #1e40af; }
    .metric-card {
        background-color: #ffffff; padding: 1rem; border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
    }
    .metric-card h3 { font-size: 1.25rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem; }
    .metric-card p { font-size: 1.875rem; font-weight: 700; color: #2563eb; }
    .section-header { font-size: 1.5rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ======= Data Load =======
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)
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

    if 'Task_Status' in df.columns:
        df['Task_Status'] = df['Task_Status'].astype(str).str.upper()
    if 'Delayed' in df.columns:
        df['Delayed'] = df['Delayed'].astype(str).str.title()
    if 'Follow_Up_Status' in df.columns:
        df['Follow_Up_Status'] = df['Follow_Up_Status'].astype(str).str.title()

    return df

df = load_data()

# ======= Sidebar Filters =======
with st.sidebar:
    st.header("📁 Filters", anchor=False)

    territory_options = ["All"] + sorted(df['Territory_Name'].dropna().unique().tolist()) if 'Territory_Name' in df else ["All"]
    territory = st.selectbox("Territory", options=territory_options, key="territory")
    filtered_df = df.copy()
    if territory != "All" and 'Territory_Name' in df:
        filtered_df = filtered_df[filtered_df['Territory_Name'] == territory]

    cluster_options = ["All"] + sorted(filtered_df['CLUSTER'].dropna().unique().tolist()) if 'CLUSTER' in filtered_df else ["All"]
    cluster = st.selectbox("Cluster", options=cluster_options, key="cluster")
    if cluster != "All" and 'CLUSTER' in filtered_df:
        filtered_df = filtered_df[filtered_df['CLUSTER'] == cluster]

    branch_options = ["All"] + sorted(filtered_df['Branch'].dropna().unique().tolist()) if 'Branch' in filtered_df else ["All"]
    branch = st.selectbox("Branch", options=branch_options, key="branch")
    if branch != "All" and 'Branch' in filtered_df:
        filtered_df = filtered_df[filtered_df['Branch'] == branch]

    task_options = ["All"] + sorted(filtered_df['Task_Type'].dropna().unique().tolist()) if 'Task_Type' in filtered_df else ["All"]
    task = st.selectbox("Task Type", options=task_options, key="task_type")
    if task != "All" and 'Task_Type' in filtered_df:
        filtered_df = filtered_df[filtered_df['Task_Type'] == task]

    status_options = ["All"] + sorted(filtered_df['Task_Status'].dropna().unique().tolist()) if 'Task_Status' in filtered_df else ["All"]
    status = st.selectbox("Status", options=status_options, key="status")
    if status != "All" and 'Task_Status' in filtered_df:
        filtered_df = filtered_df[filtered_df['Task_Status'] == status]

    designation_options = ["All"] + sorted(filtered_df['Designation'].dropna().unique().tolist()) if 'Designation' in filtered_df else ["All"]
    designation = st.selectbox("Designation", options=designation_options, key="designation")
    if designation != "All" and 'Designation' in filtered_df:
        filtered_df = filtered_df[filtered_df['Designation'] == designation]

    zone_options = ["All"] + sorted(filtered_df['Zone'].dropna().unique().tolist()) if 'Zone' in filtered_df else ["All"]
    zone = st.selectbox("Zone", options=zone_options, key="zone")
    if zone != "All" and 'Zone' in filtered_df:
        filtered_df = filtered_df[filtered_df['Zone'] == zone]

    delayed_options = ["All"] + sorted(filtered_df['Delayed'].dropna().unique().tolist()) if 'Delayed' in filtered_df else ["All"]
    delayed = st.selectbox("Delayed", options=delayed_options, key="delayed")
    if delayed != "All" and 'Delayed' in filtered_df:
        filtered_df = filtered_df[filtered_df['Delayed'] == delayed]

    follow_up_options = ["All"] + sorted(filtered_df['Follow_Up_Status'].dropna().unique().tolist()) if 'Follow_Up_Status' in filtered_df else ["All"]
    follow_up = st.selectbox("Follow Up Status", options=follow_up_options, key="follow_up")
    if follow_up != "All" and 'Follow_Up_Status' in filtered_df:
        filtered_df = filtered_df[filtered_df['Follow_Up_Status'] == follow_up]

    df = filtered_df.copy()

# ======= KPIs =======
st.markdown("<div class='section-header'>📊 Task Analytics Dashboard</div>", unsafe_allow_html=True)
total = len(df)
comp = df['Task_Status'].eq('COMPLETED').sum() if 'Task_Status' in df else 0
pend = df['Task_Status'].eq('PENDING').sum() if 'Task_Status' in df else 0
prog = df['Task_Status'].eq('IN PROGRESS').sum() if 'Task_Status' in df else 0
delayed_count = df['Delayed'].eq('Yes').sum() if 'Delayed' in df else 0
avg_accuracy = df['Accuracy_Percentage'].mean().round(1) if 'Accuracy_Percentage' in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(f"""<div class='metric-card'><h3>Total Tasks</h3><p>{total}</p></div>""", unsafe_allow_html=True)
with k2: st.markdown(f"""<div class='metric-card'><h3>✅ Completed</h3><p>{comp}</p></div>""", unsafe_allow_html=True)
with k3: st.markdown(f"""<div class='metric-card'><h3>🕒 In Progress</h3><p>{prog}</p></div>""", unsafe_allow_html=True)
with k4: st.markdown(f"""<div class='metric-card'><h3>⏳ Pending</h3><p>{pend}</p></div>""", unsafe_allow_html=True)
with k5: st.markdown(f"""<div class='metric-card'><h3>🚨 Delayed</h3><p>{delayed_count}</p></div>""", unsafe_allow_html=True)
st.markdown("---")

# ======= Territory × Task Type =======
st.markdown("<div class='section-header'>📌 Territory × Task Type Matrix</div>", unsafe_allow_html=True)
if 'Territory_Name' in df.columns and 'Task_Type' in df.columns:
    terr_type_pivot = pd.pivot_table(
        df, index='Territory_Name', columns='Task_Type',
        values='Task_ID' if 'Task_ID' in df.columns else df.columns[0],
        aggfunc='count', fill_value=0, margins=False
    )
    total_row = pd.DataFrame(terr_type_pivot.sum(axis=0)).T; total_row.index = ['Total']
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

# ======= Territory, Cluster, Status =======
st.markdown("<div class='section-header'>🧮 Performance by Territory, Cluster & Status</div>", unsafe_allow_html=True)
if set(['Territory_Name', 'CLUSTER', 'Task_Status']).issubset(df.columns):
    perf_pivot = pd.pivot_table(
        df, index=['Territory_Name', 'CLUSTER'], columns='Task_Status',
        values='Task_ID' if 'Task_ID' in df.columns else df.columns[0],
        aggfunc='count', fill_value=0, margins=False
    ).reset_index()

    for status in ['COMPLETED', 'IN PROGRESS', 'PENDING']:
        if status not in perf_pivot.columns: perf_pivot[status] = 0

    status_cols = [c for c in perf_pivot.columns if c not in ['Territory_Name', 'CLUSTER']]
    perf_pivot['GRAND TOTAL'] = perf_pivot[status_cols].sum(axis=1)
    perf_pivot['% COMPLETED'] = (perf_pivot['COMPLETED'] / perf_pivot['GRAND TOTAL'] * 100).round(1).fillna(0)

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

# ======= Cluster-Branch table with territory filter =======
territory_values = ['All'] + sorted(df['Territory_Name'].dropna().unique().tolist()) if 'Territory_Name' in df else ['All']
selected_territory = st.selectbox("Filter by Territory (for Cluster-Branch Table)", territory_values, key="clusterbranch_territory")
filtered_df_cb = df[df['Territory_Name'] == selected_territory] if selected_territory != 'All' and 'Territory_Name' in df else df.copy()

st.markdown("<div class='section-header'>🗂️ Cluster, Branch, Task Status Performance</div>", unsafe_allow_html=True)
if set(['CLUSTER', 'Branch', 'Task_Status']).issubset(filtered_df_cb.columns):
    cb_pivot = pd.pivot_table(
        filtered_df_cb, index=['CLUSTER', 'Branch'], columns='Task_Status',
        values='Task_ID' if 'Task_ID' in filtered_df_cb.columns else filtered_df_cb.columns[0],
        aggfunc='count', fill_value=0
    ).reset_index()

    for status in ['COMPLETED', 'IN PROGRESS', 'PENDING']:
        if status not in cb_pivot.columns: cb_pivot[status] = 0

    status_cols = [c for c in cb_pivot.columns if c not in ['CLUSTER', 'Branch']]
    cb_pivot['TOTAL'] = cb_pivot[status_cols].sum(axis=1)
    cb_pivot['% COMPLETED'] = (cb_pivot['COMPLETED'] / cb_pivot['TOTAL'] * 100).round(1).fillna(0)

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

# ======= Charts =======
c1, c2 = st.columns(2)
with c1:
    pie_df = pd.DataFrame({'Status': ['Completed', 'In Progress', 'Pending'], 'Count': [comp, prog, pend]})
    fig = px.pie(pie_df, values='Count', names='Status', hole=0.3,
                 color_discrete_sequence=['#1e3a8a', '#8b5cf6', '#e11d48'])
    fig.update_layout(title="Task Status Distribution", title_x=0.5, font=dict(size=14),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    if 'Task_Type' in df.columns and 'Task_Status' in df.columns:
        bar = df.groupby(['Task_Type', 'Task_Status']).size().reset_index(name='Count')
        fig2 = px.bar(bar, x='Task_Type', y='Count', color='Task_Status', barmode='stack',
                      color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6'])
        fig2.update_layout(title="Tasks by Type and Status", title_x=0.5, xaxis_title="Task Type",
                           yaxis_title="Count", xaxis_tickangle=-45, font=dict(size=14))
        st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

# ======= Accuracy Distribution =======
if 'Accuracy_Percentage' in df.columns:
    st.markdown("<div class='section-header'>📏 Accuracy Distribution</div>", unsafe_allow_html=True)
    fig_accuracy = px.histogram(df, x='Accuracy_Percentage', nbins=20, title="Distribution of Task Accuracy",
                                color_discrete_sequence=['#2563eb'])
    fig_accuracy.update_layout(title_x=0.5, xaxis_title="Accuracy Percentage", yaxis_title="Count", font=dict(size=14))
    st.plotly_chart(fig_accuracy, use_container_width=True)
    st.markdown("---")

# ======= Delayed Tasks =======
if 'Delayed' in df.columns and 'Task_Type' in df.columns:
    st.markdown("<div class='section-header'>⏰ Delayed Tasks Analysis</div>", unsafe_allow_html=True)
    delayed_df = df.groupby(['Task_Type', 'Delayed']).size().reset_index(name='Count')
    fig_delayed = px.bar(delayed_df, x='Task_Type', y='Count', color='Delayed', barmode='group',
                         color_discrete_sequence=['#ef4444', '#22c55e'])
    fig_delayed.update_layout(title="Delayed vs Non-Delayed Tasks by Type", title_x=0.5,
                              xaxis_title="Task Type", yaxis_title="Count", xaxis_tickangle=-45, font=dict(size=14))
    st.plotly_chart(fig_delayed, use_container_width=True)
    st.markdown("---")

# ======= Task Location Analysis (Employee-wise, Count-based) =======
st.markdown("<div class='section-header'>🗺️ Task Location Analysis (Employee-wise Repeats)</div>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    _default_radius = int(globals().get("CLUSTER_RADIUS_METERS", 500))
    RADIUS_M = st.number_input(
        "DBSCAN radius (meters)", min_value=100, max_value=3000,
        value=_default_radius, step=50, key="dbscan_radius_m_empwise"
    )
    REPEAT_MIN_COUNT = st.number_input(
        "Minimum repeats (per employee × location)", min_value=2, max_value=50,
        value=2, step=1, key="repeat_min_count_empwise"
    )
    PERF_FILTER_REPEATS = st.checkbox(
        "Filter performance tables by repeated-location employees only",
        value=False, key="perf_filter_repeats"
    )

required_cols = ['Latitude', 'Longitude']
repeated_details = pd.DataFrame()
if set(required_cols).issubset(df.columns):
    map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()

    if map_df.empty:
        st.success("✅ No GPS data available.")
    else:
        # Normalize critical columns
        if 'Employee' not in map_df.columns:
            map_df['Employee'] = 'Unknown'
        map_df['Employee'] = map_df['Employee'].fillna('Unknown').astype(str)

        # Ensure Task_ID exists
        if 'Task_ID' not in map_df.columns:
            map_df['Task_ID'] = map_df.index.astype(str)

        # Ensure Mobile_Number exists (if your column is named differently, rename here)
        if 'Mobile_Number' not in map_df.columns:
            map_df['Mobile_Number'] = None
        map_df['Mobile_Number'] = map_df['Mobile_Number'].astype(str).replace({'nan': None})

        # DBSCAN on all points (haversine distance)
        coords = map_df[['Latitude', 'Longitude']].values
        coords_rad = np.radians(coords)
        db_map = DBSCAN(
            eps=RADIUS_M / 6371000,  # meters -> radians
            min_samples=2,
            algorithm='ball_tree',
            metric='haversine'
        )
        clusters_map = db_map.fit_predict(coords_rad)
        map_df['Cluster'] = clusters_map

        clustered = map_df[map_df['Cluster'] != -1].copy()  # exclude noise

        if clustered.empty:
            st.success(f"✅ No same-location clusters detected within {RADIUS_M}m for any employee.")
        else:
            # helpers
            def mode_or_none(s):
                try:
                    m = s.dropna().mode()
                    return m.iloc[0] if not m.empty else None
                except Exception:
                    return None

            def first_non_null(s):
                s = s.dropna()
                return s.iloc[0] if not s.empty else None

            # ===== Aggregate to Employee × Cluster, including Mobile_Number =====
            details = (
                clustered
                .groupby(['Employee', 'Cluster'], as_index=False)
                .agg(
                    Mobile_Number=('Mobile_Number', first_non_null),  # <-- added
                    Count=('Task_ID', 'count'),
                    Latitude=('Latitude', 'mean'),
                    Longitude=('Longitude', 'mean'),
                    Branch=('Branch', mode_or_none),
                    Territory_Name=('Territory_Name', mode_or_none),
                    CLUSTER=('CLUSTER', mode_or_none),
                    Task_Status_Sample=('Task_Status', mode_or_none),
                    Task_IDs=('Task_ID', lambda s: ', '.join(map(str, s.tolist())))
                )
            )

            # Keep only repeated same-location cases per employee
            repeated_details = details[details['Count'] >= REPEAT_MIN_COUNT].copy()

            if repeated_details.empty:
                st.success(
                    f"✅ No repeated same-location tasks (per employee) found within {RADIUS_M}m "
                    f"(threshold ≥ {REPEAT_MIN_COUNT})."
                )
            else:
                # ===== Employee Summary =====
                emp_summary = (
                    repeated_details
                    .groupby(['Employee', 'Mobile_Number'], as_index=False)
                    .agg(
                        Repeated_Locations=('Cluster', 'nunique'),
                        Total_Tasks_at_Repeated_Locations=('Count', 'sum'),
                        Max_Repeat_At_Single_Location=('Count', 'max')
                    )
                    .sort_values(
                        ['Total_Tasks_at_Repeated_Locations', 'Repeated_Locations', 'Max_Repeat_At_Single_Location'],
                        ascending=[False, False, False]
                    )
                )

                st.subheader("👤 Employee Summary (Repeated Same-Location Activity)")
                st.dataframe(emp_summary, use_container_width=True)
                st.download_button(
                    "Download Employee Summary (CSV)",
                    emp_summary.to_csv(index=False),
                    file_name="employee_same_location_summary.csv",
                    mime="text/csv",
                    key=f"download_emp_summary_{uuid.uuid4()}"
                )

                # ===== Detailed table (includes Mobile_Number) =====
                st.subheader("🔎 Detailed Repeats by Employee × Location Cluster")
                repeated_details = repeated_details.sort_values(
                    ['Employee', 'Count', 'Cluster'], ascending=[True, False, True]
                )
                st.dataframe(repeated_details, use_container_width=True)
                st.download_button(
                    "Download Detailed Repeats (CSV)",
                    repeated_details.to_csv(index=False),
                    file_name="employee_same_location_details.csv",
                    mime="text/csv",
                    key=f"download_emp_details_{uuid.uuid4()}"
                )

                # ===== Aggregated map (one bubble per Employee × Cluster) =====
                st.subheader("🗺️ Map: Repeated Same-Location Bubbles (per Employee)")
                center_lat = repeated_details['Latitude'].mean()
                center_lon = repeated_details['Longitude'].mean()

                fig_map_agg = px.scatter_mapbox(
                    repeated_details,
                    lat="Latitude",
                    lon="Longitude",
                    size="Count",
                    hover_name="Employee",
                    hover_data={
                        "Mobile_Number": True,  # <-- surfaced in hover
                        "Count": True,
                        "Cluster": True,
                        "Branch": True,
                        "Territory_Name": True,
                        "CLUSTER": True,
                        "Task_Status_Sample": True,
                        "Task_IDs": False  # long; kept in table/export
                    },
                    zoom=6,
                    height=600
                )
                fig_map_agg.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_center={"lat": center_lat, "lon": center_lon},
                    margin=dict(l=0, r=0, t=0, b=0),
                    title=f"Repeated Same-Location Tasks (radius={RADIUS_M}m, bubble size = repeat count)",
                    title_x=0.5,
                    font=dict(size=14)
                )
                st.plotly_chart(fig_map_agg, use_container_width=True)

                st.warning(
                    f"🚨 Found {int(repeated_details['Count'].sum())} tasks at repeated same locations "
                    f"across {repeated_details['Employee'].nunique()} employees "
                    f"({repeated_details['Cluster'].nunique()} location clusters)."
                )
else:
    st.info("GPS columns not found.")

# ===== Build inclusion/exclusion sets for Performance sections =====
# Employees involved in repeated same-location tasks
if not repeated_details.empty:
    repeated_employee_set = set(repeated_details['Employee'].unique())
else:
    repeated_employee_set = set()

# Decide which employees to include in performance tables
if 'Employee' in df.columns:
    all_employees_set = set(df['Employee'].dropna().astype(str).unique())
else:
    all_employees_set = set()

# Sidebar toggle: include only repeated-location employees or everyone
if 'PERF_FILTER_REPEATS' in st.session_state and st.session_state['perf_filter_repeats']:
    valid_employees = repeated_employee_set
else:
    valid_employees = all_employees_set

# Branches that contain at least one repeated same-location task (via those employees)
if 'Employee' in df.columns and 'Branch' in df.columns:
    repeat_branch_set = set(
        df[df['Employee'].astype(str).isin(repeated_employee_set)]['Branch']
        .dropna().astype(str).unique()
    )
else:
    repeat_branch_set = set()

st.markdown("---")

# ======= Performance Analysis =======
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

    # Constrain by valid employees (based on sidebar toggle)
    if len(valid_employees) > 0:
        emp_task_filt = emp_task_filt[emp_task_filt['Employee'].astype(str).isin(valid_employees)]

    # BEST: exclude employees with repeated same-location tasks
    best_pool = emp_task_filt[~emp_task_filt['Employee'].astype(str).isin(repeated_employee_set)]
    best_emps = best_pool.sort_values(['Completion_%', 'Avg_Accuracy', 'Completed'], ascending=[False, False, False]).head(5)

    # POOR: unchanged (can include repeated-location employees)
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

    # ======= Branch Level =======
    br_cols = ['Branch', 'CLUSTER', 'Territory_Name', 'Zone']
    if all(c in df.columns for c in br_cols):
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

        # BEST BRANCHES: exclude branches with any repeated same-location task
        best_branch_pool = branch_task_filt[~branch_task_filt['Branch'].astype(str).isin(repeat_branch_set)]
        best_branch = best_branch_pool.sort_values(['Completion_%', 'Avg_Accuracy', 'Completed'], ascending=[False, False, False]).head(3)

        # POOR BRANCHES: unchanged
        poor_branch = branch_task_filt.sort_values(['Completion_%', 'Delayed_Tasks', 'Total_Tasks'], ascending=[True, False, False]).head(3)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏢 Top 3 Best Branches")
            st.dataframe(best_branch, use_container_width=True)
        with c2:
            st.subheader("🚩 Bottom 3 Branches")
            st.dataframe(poor_branch, use_container_width=True)
st.markdown("---")

# ======= Cluster-wise Status Charts =======
if 'CLUSTER' in df.columns and 'Task_Status' in df.columns:
    st.markdown("<div class='section-header'>🏢 Cluster-wise Analysis</div>", unsafe_allow_html=True)
    cluster_status = df.groupby(['CLUSTER', 'Task_Status']).size().reset_index(name='Count')
    fig_cbar = px.bar(cluster_status, x='CLUSTER', y='Count', color='Task_Status',
                      barmode='stack', title='Tasks by Status per Cluster',
                      color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6'])
    fig_cbar.update_layout(title_x=0.5, xaxis_tickangle=-45, font=dict(size=14))
    st.plotly_chart(fig_cbar, use_container_width=True)
    st.dataframe(cluster_status.pivot(index='CLUSTER', columns='Task_Status', values='Count').fillna(0), use_container_width=True)
st.markdown("---")

# ======= Date Trend =======
if 'Task_Start' in df.columns:
    st.markdown("<div class='section-header'>📈 Task Trend Over Time</div>", unsafe_allow_html=True)
    df['Task_Date'] = pd.to_datetime(df['Task_Start'], errors='coerce').dt.date
    valid_trend = df.dropna(subset=['Task_Date', 'Task_Status']) if 'Task_Status' in df else pd.DataFrame()
    if not valid_trend.empty:
        trend = (
            valid_trend.groupby(['Task_Date', 'Task_Status'])
            .size().reset_index(name='Count').sort_values('Task_Date')
        )
        if len(trend) >= 1:
            fig3 = px.line(trend, x='Task_Date', y='Count', color='Task_Status',
                           title="Task Trend Over Time",
                           color_discrete_sequence=['#22c55e', '#f97316', '#3b82f6'])
            fig3.update_traces(mode='lines+markers')
            fig3.update_layout(title_x=0.5, font=dict(size=14), hovermode="x unified")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data available for trend chart.")
    else:
        st.info("No valid dates found in Task_Start for trend chart.")
else:
    st.warning("Task_Start column is missing from data.")
st.markdown("---")

# ======= Custom Pivot Lab =======
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
                df, index=index_col, columns=column_col, values=value_col,
                aggfunc=agg_func, fill_value=0
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
                fig = px.bar(plot_df, x=index_col, y='Value', color=column_col, barmode='group',
                             color_discrete_sequence=px.colors.qualitative.Plotly)
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Heatmap':
                fig = px.imshow(pivot_table, labels=dict(x=column_col, y=index_col, color=value_col),
                                color_continuous_scale='Viridis')
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == 'Pie':
                pie_data = pivot_table.sum(axis=1).reset_index(name='Value')
                fig = px.pie(pie_data, names=index_col, values='Value',
                             color_discrete_sequence=px.colors.qualitative.Plotly)
                fig.update_layout(title_x=0.5, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not create pivot: {e}")

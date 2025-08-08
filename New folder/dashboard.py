import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="📊 Task ML Dashboard", layout="wide")

# === 1. LOAD DATA ===
df = pd.read_csv(r"C:\Users\JIDHUN K M\Desktop\trackolap reports\task\newwww ml.csv")

# === 2. DATE PROCESSING ===
df["Started"] = pd.to_datetime(df["Started"], errors='coerce', dayfirst=True)
df["Completed"] = pd.to_datetime(df["Completed"], errors='coerce', dayfirst=True)

# === 3. SIDEBAR FILTERS ===
st.sidebar.header("Filters")

def default_opts(col):
    return sorted([v for v in df[col].dropna().unique() if str(v).strip() != ''])

wl_filter = st.sidebar.multiselect("Work Location", default_opts("Work Location"), default=default_opts("Work Location"))
terr_filter = st.sidebar.multiselect("Territory Name", default_opts("TERRITORY NAME"), default=default_opts("TERRITORY NAME"))
cluster_filter = st.sidebar.multiselect("Cluster", default_opts("CLUSTER"), default=default_opts("CLUSTER"))
branch_filter = st.sidebar.multiselect("Branch ID", default_opts("BRANCH ID"), default=default_opts("BRANCH ID"))
emp_filter = st.sidebar.multiselect("Employee", default_opts("Employee"), default=default_opts("Employee"))

# Date range
date_min = df["Started"].min()
date_max = df["Started"].max()
date_range = st.sidebar.date_input("Date Range", [date_min, date_max])

# === 4. APPLY FILTERS ===
df_filt = df[
    (df["Work Location"].isin(wl_filter)) &
    (df["TERRITORY NAME"].isin(terr_filter)) &
    (df["CLUSTER"].isin(cluster_filter)) &
    (df["BRANCH ID"].isin(branch_filter)) &
    (df["Employee"].isin(emp_filter)) &
    (df["Started"] >= pd.to_datetime(date_range[0])) &
    (df["Started"] <= pd.to_datetime(date_range[1]))
].copy()

st.title("📊 Task Dashboard: Management Insights")

if len(df_filt) == 0:
    st.error("No tasks match the selected filter. Please adjust your filters for data.")
    st.stop()

# === KPI Summary ===
total_tasks = len(df_filt)
completed_tasks = (df_filt["Status"].str.upper() == "COMPLETED").sum()
delayed_tasks = (df_filt["Delayed"].str.upper() == "YES").sum()
percent_completed = 100 * completed_tasks / total_tasks
percent_delayed = 100 * delayed_tasks / total_tasks

st.markdown(f"""
### Executive Summary  
**Total Tasks:** {total_tasks}  
**Completed:** {completed_tasks} ({percent_completed:.1f}%)  
**Delayed:** {delayed_tasks} ({percent_delayed:.1f}%)  
""")

# Delay Notification
if percent_delayed > 20:
    st.warning(f"⚠️ High delay rate: {percent_delayed:.1f}% of tasks delayed!")
elif percent_delayed > 0:
    st.info(f"ℹ️ Some delays detected: {percent_delayed:.1f}% of tasks.")
else:
    st.success("✅ All tasks completed on time for selected filters.")

# === Forecast Section ===
st.header("📈 Task Load Forecast (Next 7 Days)")
if df_filt["Started"].notnull().sum() > 0:
    ts = df_filt.groupby(df_filt["Started"].dt.date).size().reset_index(name="y")
    ts.rename(columns={"Started": "ds"}, inplace=True)
    ts["ds"] = pd.to_datetime(ts["ds"])
    if len(ts) > 2:
        m = Prophet()
        m.fit(ts)
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        actual_dates = ts["ds"].max()
        forecast["Type"] = np.where(forecast["ds"] <= actual_dates, "Actual", "Forecast")

        fig = px.line(forecast, x="ds", y="yhat", color="Type", line_dash="Type")
        fig.add_scatter(x=ts["ds"], y=ts["y"], mode='markers', name='Actual Data', marker=dict(color='green'))
        st.plotly_chart(fig, use_container_width=True)

        forecast_next = forecast[forecast["ds"] > actual_dates][["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted Task Count"})
        st.dataframe(forecast_next, use_container_width=True)
    else:
        st.info("Not enough time series data for forecast.")

# === ML Classification Section ===
st.header("🧠 Task Delay Prediction")
ml_features = ["Priority", "Status", "Type"]
if all(col in df_filt.columns for col in ml_features) and df_filt.shape[0] > 10:
    X = pd.get_dummies(df_filt[ml_features])
    y = (df_filt["Delayed"].str.upper() == "YES").astype(int)
    if y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model_option = st.selectbox("Choose ML Model for Delay Prediction", ["Random Forest", "Logistic Regression", "XGBoost"])
        if model_option == "Random Forest":
            clf = RandomForestClassifier()
        elif model_option == "Logistic Regression":
            clf = LogisticRegression()
        else:
            clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write("Model Accuracy:", f"{acc*100:.2f}%")

        col_pred = st.columns(3)
        priority = col_pred[0].selectbox("Priority", df_filt["Priority"].dropna().unique())
        status = col_pred[1].selectbox("Status", df_filt["Status"].dropna().unique())
        type_ = col_pred[2].selectbox("Type", df_filt["Type"].dropna().unique())

        pred_X = pd.get_dummies(pd.DataFrame([[priority, status, type_]], columns=ml_features))
        pred_X = pred_X.reindex(columns=X.columns, fill_value=0)
        pred = clf.predict(pred_X)[0]
        st.success(f"Predicted: {'Delayed' if pred else 'On Time'}")

        # Time Prediction
        st.header("⏱️ Predict Task Completion Time")
        if "Time Taken" in df_filt.columns:
            df_filt["task_seconds"] = pd.to_timedelta(df_filt["Time Taken"], errors='coerce').dt.total_seconds()
            y_time = df_filt["task_seconds"].fillna(df_filt["task_seconds"].mean())
            Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_time, test_size=0.2)

            model_option_time = st.selectbox("Choose Model for Time Prediction", ["Random Forest", "Linear Regression", "XGBoost"])
            if model_option_time == "Random Forest":
                reg = RandomForestRegressor()
            elif model_option_time == "Linear Regression":
                reg = LinearRegression()
            else:
                reg = XGBRegressor()

            reg.fit(Xr_train, yr_train)
            pred_time = reg.predict(pred_X)[0]
            st.info(f"Predicted Completion Time: {int(pred_time//60)} min {int(pred_time%60)} sec")

        # Accuracy Prediction
        st.header("🎯 Predict Task Accuracy %")
        if "Accuracy %" in df_filt.columns and df_filt["Accuracy %"].notnull().sum() > 10:
            y_acc = df_filt["Accuracy %"].fillna(df_filt["Accuracy %"].mean())
            Xacc_train, Xacc_test, yacc_train, yacc_test = train_test_split(X, y_acc, test_size=0.2)

            model_option_acc = st.selectbox("Choose Model for Accuracy % Prediction", ["Random Forest", "Linear Regression", "XGBoost"])
            if model_option_acc == "Random Forest":
                reg_acc = RandomForestRegressor()
            elif model_option_acc == "Linear Regression":
                reg_acc = LinearRegression()
            else:
                reg_acc = XGBRegressor()

            reg_acc.fit(Xacc_train, yacc_train)
            pred_acc = reg_acc.predict(pred_X)[0]
            st.info(f"Predicted Task Accuracy %: {pred_acc:.1f}%")
    else:
        st.warning("Not enough variation in 'Delayed' column for prediction.")
else:
    st.info("Need at least 10 tasks with all required columns for ML prediction.")

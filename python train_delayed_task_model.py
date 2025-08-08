import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Load and preprocess your data
df = pd.read_csv("live_data.csv")
df.columns = df.columns.str.strip()
df['Delayed'] = df['Delayed'].map({'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0, 1: 1, 0: 0})

# Standardize column names for ML
rename_map = {
    'Task_Type': 'Task',
    'Time_Taken': 'Time Taken',
    'Accuracy_Percentage': 'Accuracy %'
}
df.rename(columns=rename_map, inplace=True)


for col in ['Branch', 'CLUSTER', 'Task']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

def time_to_min(t):
    try:
        h, m, s = map(int, str(t).split(":"))
        return h*60 + m + s/60
    except:
        return 0
df['TimeTakenMin'] = df['Time Taken'].apply(time_to_min)
df['Accuracy %'] = pd.to_numeric(df['Accuracy %'], errors='coerce').fillna(0)

feature_cols = ['Branch', 'CLUSTER', 'Task', 'TimeTakenMin', 'Accuracy %']
X = df[feature_cols].fillna(0)
y = df['Delayed'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_delayed_task_model.pkl")

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
joblib.dump(lr, "lr_delayed_task_model.pkl")

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgb_delayed_task_model.pkl")

print("All models saved.")

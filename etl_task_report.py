import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
input_file = r"C:\Users\JIDHUN K M\Desktop\6branch\New folder\6 BRANCH TASK 1HOUR.xlsx"
output_file = 'live_data.csv'
CREATED_BY_FILTER = [
    "ALBIN GEORGE",
    "JUSTIN KURIAN",
    "JIDHUN K MADHU",
    "SHALU S",
    "BCoperations"
]

# --- READ EXCEL, SKIP HEADER ROWS ---
df = pd.read_excel(input_file, skiprows=9)

# --- RENAME COLUMNS FOR CONSISTENCY ---
df.rename(columns={
    'Identifier': 'Identifier',
    'Work Location': 'Branch',
    'Designation': 'Designation',
    'CLUSTER': 'CLUSTER',
    'TERRITORY NAME': 'Territory_Name',
    'Mobile': 'Mobile_Number',
    'ZONE': 'Zone',
    'BRANCH ID': 'Branch_ID',
    'Task Id': 'Task_ID',
    'Type': 'Task_Type',
    'Status': 'Task_Status',
    'Task Lat/Lng': 'Task_Lat_Lng',
    'Completed Time': 'Completed_Time',
    'Delayed': 'Delayed',
    'Follow Up': 'Follow_Up_Status',
    'Frequency': 'Task_Frequency',
    'Time Taken': 'Time_Taken',
    'Accuracy %': 'Accuracy_Percentage',
    'Start': 'Task_Start',
    'Created By': 'Created_By',
    'Title': 'Title',
    'Created': 'Created_Date'
}, inplace=True)

# --- FILTER BY 'Created_By' LIST ---
df = df[df['Created_By'].isin(CREATED_BY_FILTER)]

# --- ASK FOR DATE TO FILTER ---
date_input = input("Enter date to filter (YYYY-MM-DD) or leave blank for all: ").strip()
if date_input:
    try:
        filter_date = pd.to_datetime(date_input)
        df['Task_Start'] = pd.to_datetime(df['Task_Start'], errors='coerce')  # robust conversion
        df = df[df['Task_Start'].dt.date == filter_date.date()]
        print(f"✅ Filtered data for Task_Start date: {filter_date.date()}")
    except Exception as e:
        print(f"❌ Invalid date format. Proceeding without date filter. Error: {e}")

# --- ADD ETL TIMESTAMP COLUMNS ---
now = datetime.now()
df['ETL_Run_Date'] = now.strftime('%Y-%m-%d')
df['ETL_Run_Time'] = now.strftime('%H:%M:%S')

# --- EXPORT TO CSV ---
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ ETL complete. File saved as: {output_file}")

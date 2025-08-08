import pandas as pd

# Sample: Load your real data here
df = pd.read_csv("C:\Users\JIDHUN K M\Desktop\6branch\live_data.csv")

# If Branch Name is not available, use Branch_ID or map it using a lookup table

# Step 1: Filter for Western Maharashtra if needed
# df = df[df['CLUSTER'] == 'KOLHAPUR']  # or use your real cluster names/logic

# Step 2: Pivot table for status counts
pivot = pd.pivot_table(
    df,
    index=['CLUSTER', 'Branch_ID'],  # Use 'Branch' if available
    columns='Task_Status',
    values='Task_ID',  # or any column with unique task identifier
    aggfunc='count',
    fill_value=0
).reset_index()

# Step 3: Add Grand Total and Percentage
pivot['Grand Total'] = pivot[['COMPLETED', 'INPROGRESS', 'PENDING']].sum(axis=1)
pivot['%'] = (pivot['COMPLETED'] / pivot['Grand Total']).fillna(0).apply(lambda x: "{:.0%}".format(x))

# Step 4: Clean column names for Excel-style
pivot = pivot.rename(columns={'Branch_ID': 'Branch'})

# Step 5: Optional - Sort and display
pivot = pivot[['CLUSTER', 'Branch', 'COMPLETED', 'INPROGRESS', 'PENDING', 'Grand Total', '%']]
print(pivot)

# To Excel
 pivot.to_excel("western_maharashtra_task_status.xlsx", index=False)

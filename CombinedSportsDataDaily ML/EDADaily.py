#Purpose to conduct exploratory Data Analysis 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Techniques
#   1) Descriptive Statistics
#   2) Data Visualization
#   3) Univariate Analysis
#   4) Bivariate and Multivariate Analysis
#   5) Outlier Detection
#   6) Data Cleaning

# create directory to store images, 
output_dir = 'EDA_graphs'   # in case we went to change the name later
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv('timeseries (daily).csv')
print('\n---------------------------------- Basic Information for Competitive runners Daily ---------------------------------')

#------------------------------ Basic Information ----------------------------------------------
print("======== Shape of Data: =========")
print(df.shape,'\n')

print('========= Data Col: ==========')
print(df.columns,'\n')

print('======== Data Types: ========')
print(df.dtypes,'\n')

print('======== Head of Data: =========')
print(df.head(),'\n')

print('======== Missing Values: ========')
print(df.isnull().sum(),'\n')

print('======= Number of Duplicate Rows: ========')
print(df.duplicated().sum(), '\n')

print('======== Unique Values for Athlete ID: ========')
print(df['Athlete ID'].nunique(),'\n')

# List of features from the journal
journal_features = [
    'km Z5-T1-T2.6',
    'perceived trainingSuccess.6',
    'perceived recovery',
    'perceived exertion.4',
    'perceived exertion.3',
    'km Z5-T1-T2.4',
    'km sprinting.5',
    'nr. sessions.5',
    'strength training.6',
    'km Z3-4.1',
    'perceived recovery.1',
    'nr. sessions.2',
    'km Z5-T1-T2.5',
    'km Z3-4.3',
    'perceived trainingSuccess.4',
    'total km.1',
    'hours alternative.4',
    'perceived recovery.6',
    'perceived recovery.3',
    'hours alternative.6'
]

all_features = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery',
                'nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 'perceived exertion.1', 'perceived trainingSuccess.1', 'perceived recovery.1', 
                'nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2', 'perceived exertion.2', 'perceived trainingSuccess.2', 'perceived recovery.2', 
                'nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3', 'perceived exertion.3', 'perceived trainingSuccess.3', 'perceived recovery.3', 
                'nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4', 'perceived exertion.4', 'perceived trainingSuccess.4', 'perceived recovery.4', 
                'nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5', 'perceived exertion.5', 'perceived trainingSuccess.5', 'perceived recovery.5', 
                'nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6', 'perceived exertion.6', 'perceived trainingSuccess.6', 'perceived recovery.6',
                'Athlete ID', 'injury', 'Date']

# --Descriptive Statistics of all features ---
print('Descriptive Statistics for all features\n')
all_features_stats = df[all_features].describe().transpose()
#               .describe()     count, mean, std, min, 25/50/75 %, max
#               .tranpose()     swap row&col b/c large number of features
print(all_features_stats.to_markdown(numalign="left", stralign="left"))

print("\n--------------------------------------------------------------\n")

# --- Descriptive Statistics for journal features --
print('Descriptive Statistics for journal features\n')
journal_features_stats = df[journal_features].describe().transpose()
print(journal_features_stats.to_markdown(numalign="left", stralign="left"))

# Count Zero Ocuurences of the number '0' in the 'Date' column
zero_count = (df['Date'] == 0).sum()
print(f"\nthe number of zeros in Date col is: {zero_count}")

# count number of rows belonging to each athlete
athlete_counts = df['Athlete ID'].value_counts().to_markdown()
print("Number rows per Athlete: ")
print(athlete_counts)

# athlete-level summary statistics or aggregated features
# individual athlete info : injury rate , km mean/std , exertion mean/std  
athlete_stats = df.groupby('Athlete ID').agg({
    'injury': 'mean',
    'total km': ['mean', 'std'],
    'perceived exertion': ['mean', 'std']
})
athlete_stats.columns = ['injury_rate', 'km_mean', 'km_std', 'exertion_mean', 'exertion_std']
athlete_stats.sort_values('injury_rate', ascending=False)

#count plot of  target (injury v non inury)
sns.countplot(x='injury', data=df)
plt.title('Injury Class Distribution')
plt.xlabel('Injury (0 = No, 1 = Yes)')
plt.ylabel('Count')
#plt.show()
plt.savefig(os.path.join(output_dir, 'Injury_Class_Distribution_Countplot.png'))

# Extract day 0 features (no dot in the name)
#error b/c feature 'nr. sessions' contains dot and is unintentionally ommitted6
day0_cols = [col for col in df.columns if '.' not in col and col not in ['Athlete ID', 'injury', 'Date']]
day0_corr = df[day0_cols].corr()

#feature columns 
day0_features = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery', 'injury']
day1_features = ['nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 'perceived exertion.1', 'perceived trainingSuccess.1', 'perceived recovery.1', 'injury']
day2_features = ['nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2', 'perceived exertion.2', 'perceived trainingSuccess.2', 'perceived recovery.2', 'injury']
day3_features = ['nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3', 'perceived exertion.3', 'perceived trainingSuccess.3', 'perceived recovery.3', 'injury']
day4_features = ['nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4', 'perceived exertion.4', 'perceived trainingSuccess.4', 'perceived recovery.4', 'injury']
day5_features = ['nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5', 'perceived exertion.5', 'perceived trainingSuccess.5', 'perceived recovery.5', 'injury']
day6_features = ['nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6', 'perceived exertion.6', 'perceived trainingSuccess.6', 'perceived recovery.6', 'injury']

#feature correlation matrices
day0_corr2 = df[day0_features].corr()
day1_corr = df[day1_features].corr()
day2_corr = df[day2_features].corr()
day3_corr = df[day3_features].corr()
day4_corr = df[day4_features].corr()
day5_corr = df[day5_features].corr()
day6_corr = df[day6_features].corr()

#heat map to compare features heat maps 0-6
plt.figure(figsize=(12, 10))
sns.heatmap(day0_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 0 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_0_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day0_corr2, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 0 (fixed) Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_0_Fixed_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day1_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 1 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_1_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day2_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 2 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_2_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day3_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 3 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_3_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day4_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 4 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_4_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day5_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 5 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_5_Heatmap.png'))

plt.figure(figsize=(12, 10))
sns.heatmap(day6_corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Day 6 Features")
#plt.show()
plt.savefig(os.path.join(output_dir, 'Day_6_Heatmap.png'))

# ---------------------------- Mean Bar Comparison -----------------------------------------------
# Group by injury
grouped = df.groupby('injury')[day0_cols]

# Mean comparison
mean_comparison = grouped.mean().T
mean_comparison.plot(kind='bar', figsize=(14, 6))
plt.title('Feature Means by Injury Class (Day 0)')
plt.ylabel('Mean Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Feature_Means_By_Injury_Class_Day0_Bar.png'))

# --------------------------------------- 7 day Line Chart ----------------------------------------------
# Pick one feature to plot over 7 days
feature_base = 'total km'
feature_cols = [feature_base] + [f'{feature_base}.{i}' for i in range(1, 7)]

# Calculate average per injury class
means_over_time = df.groupby('injury')[feature_cols].mean().T
means_over_time.columns = ['No Injury', 'Injury']
means_over_time.index = ['Day 0'] + [f'Day -{i}' for i in range(1, 7)]

means_over_time.plot(marker='o', figsize=(10, 5))
plt.title(f'Average {feature_base} Over 7 Days by Injury Class')
plt.ylabel('Avg Value')
plt.xlabel('Day')
plt.gca().invert_xaxis()
plt.grid(True)
#plt.show()
plt.savefig(os.path.join(output_dir, 'Average_Over_7Days_Chart.png'))

# -------------------------------- line chart with multiple features ----------------------------------------------------------

features_to_plot = ['total km', 'nr. sessions', 'perceived exertion', 'perceived recovery']
for feature in features_to_plot:
    days = [f"{feature}" if i == 0 else f"{feature}.{i}" for i in range(7)]
    df_injured = df[df['injury'] == 1][days].mean()
    df_healthy = df[df['injury'] == 0][days].mean()
    df_injured.index = df_healthy.index = [f'Day -{i}' for i in range(7)][::-1]
    plt.plot(df_injured.index, df_injured.values, marker='o', label=f'{feature} (Injury)')
    plt.plot(df_healthy.index, df_healthy.values, marker='x', linestyle='--', label=f'{feature} (No Injury)')
plt.title('Feature Trends Before Injury vs No Injury')
plt.xlabel('Day')
plt.ylabel('Mean Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Feature_Trends_Before_Injury_v_No_Injury_Boxplot.png'))

# ---------------------------------------------  box plot ------------------------------------------------------------------------------

features = ['total km', 'km Z3-4', 'km Z5-T1-T2', 'perceived exertion', 'perceived recovery']
df_melted = df[['injury'] + features].melt(id_vars='injury')
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melted, x='variable', y='value', hue='injury')
plt.title('Feature Distributions by Injury (Day 0)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
#plt.show()
plt.savefig(os.path.join(output_dir, 'Feature_Distributions_by_Injury_Day0_Boxplot.png'))

# --------------------------------------------------- box plot strain v injury ---------------------------------------------------
# create list of col names for 'total km' for each of 7 days
#   if i=0 return 'total km' , else 'total km.i'
km_cols = [f'total km' if i == 0 else f'total km.{i}' for i in range(7)]
# sports science metric monotony
df['monotony'] = df[km_cols].std(axis=1) # axis=1 will conduct the SD row wise
# sports science metric using monotony
df['strain'] = df[km_cols].mean(axis=1) * df['monotony']
sns.boxplot(x='injury', y='strain', data=df)
plt.title('Objective Strain vs Injury')
plt.grid(True)
#plt.show()
plt.savefig(os.path.join(output_dir, 'Objective_Strain_v_Injury_Boxplot.png'))

#--------------------------------------------------- box plot for RPE strain ------------------------------------------------------------

#km_cols = [f'total km' if i == 0 else f'total km.{i}' for i in range(7)] this line excluded for time being b/c included above
# Create a list of column names for the past 7 days' perceived exertion (RPE)
# needed to be able to navigate to them directly later one (the next 7 lines of code)
rpe_cols = [f'perceived exertion' if i == 0 else f'perceived exertion.{i}' for i in range(7)]

# Calculate the Session RPE for each of the last 7 days.
# We're using 'total km' as a proxy for duration since a specific duration column is not available.
df['Session RPE (d-0)'] = df['total km'] * df['perceived exertion']
df['Session RPE (d-1)'] = df['total km.1'] * df['perceived exertion.1']
df['Session RPE (d-2)'] = df['total km.2'] * df['perceived exertion.2']
df['Session RPE (d-3)'] = df['total km.3'] * df['perceived exertion.3']
df['Session RPE (d-4)'] = df['total km.4'] * df['perceived exertion.4']
df['Session RPE (d-5)'] = df['total km.5'] * df['perceived exertion.5']
df['Session RPE (d-6)'] = df['total km.6'] * df['perceived exertion.6']

# Create a list of the new Session RPE columns (creaetd right above)
session_rpe_cols = [f'Session RPE (d-{i})' for i in range(7)]

# Calculate the new monotony (standard deviation of Session RPE)
df['rpe_monotony'] = df[session_rpe_cols].std(axis=1)

# Calculate the new strain (mean of Session RPE multiplied by rpe_monotony)
df['rpe_strain'] = df[session_rpe_cols].mean(axis=1) * df['rpe_monotony']

# Remove rows with missing values that might have been created by the rolling window calculation
df_clean = df.dropna(subset=['rpe_strain', 'injury'])

# Create a box plot to visualize the relationship between the new 'rpe_strain' and 'injury'
plt.figure(figsize=(8, 6))
sns.boxplot(x='injury', y='rpe_strain', data=df_clean)
plt.title('Combined Strain vs Injury')
plt.xlabel('Injury (0 = No Injury, 1 = Injured)')
plt.ylabel('RPE Strain')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Combined_Strain_v_injury_Boxplot.png'))

#------------------------------------------- Mean feature values leading up to injury -------------------------------------------------

injury_indices = df[df['injury'] == 1].index    #find row index of every row with injury
lags = range(1, 7)          #define time window of 0-6 lagged variables

lagged_means = {}
for lag in lags:
    valid_indices = [i for i in injury_indices if i - lag >= 0]
    lagged_frames = [df.iloc[i - lag:i] for i in valid_indices]
    lagged_df = pd.concat(lagged_frames)
    lagged_means[f'Day -{lag}'] = lagged_df.mean()

injury_lagged = pd.DataFrame(lagged_means).T
injury_lagged.plot(kind='bar', figsize=(14, 6))
plt.title("Mean Feature Values Leading up to Injury Days")
plt.ylabel("Average Value")
plt.xlabel("Lag Day")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Mean_Feature_Values_Leading_To_Injury_Days_Bar.png'))

#------------------------------------ Line Charts of injured athletes total km & percieved recovery ----------------------------------------

# create numPy array 'injured_ids' that contains first 5 injured athletes
injured_ids = df[df['injury'] == 1]['Athlete ID'].unique()[:5]  # first 5 athletes with injuries df[][] = chained indexing, df['injury'] == 1 = boolean conditional, 
for aid in injured_ids:
    athlete_data = df[df['Athlete ID'] == aid].sort_values('Date') # filter id = aid, filter date of id
    fig, ax = plt.subplots(figsize=(10, 5))
    for feature in ['total km', 'perceived recovery']:  # loop through features from 'total km' to 'perc recov' (2w)
        y = athlete_data[[f"{feature}" if i == 0 else f"{feature}.{i}" for i in range(7)]].mean(axis=1)
        ax.plot(athlete_data['Date'], y, marker='o', label=feature)
    injury_dates = athlete_data[athlete_data['injury'] == 1]['Date']
    
    # Iterate through each injury date to plot a red dot on the graph
    for date in injury_dates:
        # Get the corresponding y-values for the injury date
        y_km = athlete_data[athlete_data['Date'] == date][['total km', 'total km.1', 'total km.2', 'total km.3', 'total km.4', 'total km.5', 'total km.6']].mean(axis=1).values[0]
        y_rec = athlete_data[athlete_data['Date'] == date][['perceived recovery', 'perceived recovery.1', 'perceived recovery.2', 'perceived recovery.3', 'perceived recovery.4', 'perceived recovery.5', 'perceived recovery.6']].mean(axis=1).values[0]
        
        # Plot the red dots
        ax.plot(date, y_km, 'ro', label='Injury (total km)' if 'total km' not in ax.get_legend_handles_labels()[1] else '')
        ax.plot(date, y_rec, 'ro', label='Injury (perceived recovery)' if 'perceived recovery' not in ax.get_legend_handles_labels()[1] else '')
    
    # Sets the plot title, labels, legend, and grid.
    ax.set_title(f'Athlete ID {aid} Feature Profile Around Injuries')
    ax.set_xlabel('Date')
    ax.set_ylabel('7-Day Rolling Mean')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_dir, '7_Day_Rolling_Mean_Chart.png'))

#-------------------------------------------- Objective ACWR Boxplot ---------------------------------------------------

#sort data for rolling calculation
df['Date'] = pd.to_datetime(df['Date']) #ensure datetime format  
df = df.sort_values(by=['Athlete ID', 'Date'])  #sort date by athlete id then by date

#Define the workload metric (objective) 
objective_workload = 'total km' # best practice for readability to follow the formula, allows change of metric

#Acute workload (7 day-rolling sum)
#grouby 'athlete id' to ensure metric restarts for each athlete <- this crteates multi series
#reset_index() - takes multi level index and turns back to col, level=0 is 'athlete id', drop=T - discard old index and not add as new col.
df['acute_workload'] = df.groupby('Athlete ID')[objective_workload].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)

#Chronic workload (28 rolling average)
df['chronic_workload'] = df.groupby('Athlete ID')[objective_workload].rolling(window=28, min_periods=1).sum().reset_index(level=0, drop=True)

#ACWR formula
df['ACWR'] = df['acute_workload'] / df['chronic_workload']

#show data
print('====== Data with ACWR =====')
print(df[['Date', 'Athlete ID', objective_workload, 'acute_workload', 'chronic_workload', 'ACWR', 'injury']].head(10))

# Create the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='injury', y='ACWR', data=df)
plt.title('Objective ACWR Distribution for Injured vs. Uninjured Days')
plt.xlabel('Injury Status (0 = Uninjured, 1 = Injured)')
plt.ylabel('Acute:Chronic Workload Ratio (ACWR)')
plt.xticks([0, 1], ['Uninjured', 'Injured'])
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
#plt.show() 
plt.savefig(os.path.join(output_dir, 'Objective_ACWR_Boxplot.png'))

# Find the first day of injury for each athlete
injured_df = df[df['injury'] == 1]
first_injury_df = injured_df.drop_duplicates(subset=['Athlete ID'], keep='first')

# Create the histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=first_injury_df, x='ACWR', bins=20, kde=True)
plt.title('Distribution of Objective ACWR on the First Day of Injury')
plt.xlabel('Acute:Chronic Workload Ratio (ACWR)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Objective_ACWR_First_Day_Injury_Histogram.png'))

#-------------------------------------------------  Combined ACWR -----------------------------------------------

# define workload metric with combined subjective/objective data
df['combined_workload'] = df['total km']* df['perceived exertion']

# Calculate Acute and Chronic Workload using the new metric
df['acute_workload_combined'] = df.groupby('Athlete ID')['combined_workload'].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
df['chronic_workload_combined'] = df.groupby('Athlete ID')['combined_workload'].rolling(window=28, min_periods=1).sum().reset_index(level=0, drop=True)

# Calculate the new ACWR
df['ACWR_combined'] = df['acute_workload_combined'] / df['chronic_workload_combined']

# Display the new features
print('======== Data with Combined Workload and ACWR: =========')
print(df[['Date', 'Athlete ID', 'acute_workload_combined', 'chronic_workload_combined', 'ACWR_combined']].head(10))

#-------------------------------------------------  Combined Workload Histogram---------------------------------------------------

# Find the first day of injury for each athlete using the combined ACWR
# This part is needed to filter the data correctly for the histogram
injured_df_combined = df[df['injury'] == 1]
first_injury_df_combined = injured_df_combined.drop_duplicates(subset=['Athlete ID'], keep='first')

# Create the histogram for the combined ACWR
plt.figure(figsize=(10, 6))
sns.histplot(data=first_injury_df_combined, x='ACWR_combined', bins=20, kde=True)
plt.title('Distribution of ACWR (Combined) on the First Day of Injury')
plt.xlabel('Combined Acute:Chronic Workload Ratio (ACWR)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(output_dir, 'Combined_ACWR_First_Day_Injury_Histplot.png'))
#------------------------------------------------- Combined ACWR per athlete, injured vs uninjured ---------------------------------------------------

# Find unique athlete IDs to sample from
all_athletes = df['Athlete ID'].unique()

# 1. Select 5 injured athletes
injured_athletes = df[df['injury'] == 1]['Athlete ID'].unique()
# Handle case where there are fewer than 10 injured athletes
num_injured_to_select = min(10, len(injured_athletes))
selected_injured = pd.Series(injured_athletes).sample(n=num_injured_to_select, random_state=1)
# Filter for the first injury date for these selected athletes and get their combined ACWR
injured_data_combined = df[df['Athlete ID'].isin(selected_injured) & (df['injury'] == 1)].drop_duplicates(subset=['Athlete ID'], keep='first')
injured_data_combined = injured_data_combined[['Athlete ID', 'ACWR_combined']]
injured_data_combined['Status'] = 'Injured'

# 2. Select 5 uninjured athletes
uninjured_athletes = [aid for aid in all_athletes if aid not in injured_athletes]
# Handle case where there are fewer than 10 uninjured athletes
num_uninjured_to_select = min(10, len(uninjured_athletes))
selected_uninjured = pd.Series(uninjured_athletes).sample(n=num_uninjured_to_select, random_state=1)
# Find the maximum combined ACWR for each of these uninjured athletes to represent their peak load
uninjured_data_combined = df[df['Athlete ID'].isin(selected_uninjured)].groupby('Athlete ID')['ACWR_combined'].max().reset_index()
uninjured_data_combined['Status'] = 'Uninjured'

# 3. Combine the data and create the bar plot
plot_data_combined = pd.concat([injured_data_combined, uninjured_data_combined])
plot_data_combined['Athlete ID'] = plot_data_combined['Athlete ID'].astype(str)

plt.figure(figsize=(12, 8))
sns.barplot(x='Athlete ID', y='ACWR_combined', hue='Status', data=plot_data_combined, palette={'Injured': 'red', 'Uninjured': 'blue'})
plt.title('Combined ACWR Scores for Injured vs. Uninjured Athletes')
plt.xlabel('Athlete ID')
plt.ylabel('Combined Acute:Chronic Workload Ratio (ACWR)')
plt.xticks(rotation=45)
plt.tight_layout()
#aplt.show()
plt.savefig(os.path.join(output_dir, 'Combined_ACWR_per_Athlete_Injured_v_Uninjured_Bar.png'))
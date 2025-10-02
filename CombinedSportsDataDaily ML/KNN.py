import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import altair as alt

# Load the dataset
df = pd.read_csv('timeseries (daily).csv')

# Display the first 5 rows
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Get information about the columns
print(df.info())



# Define features (X) and target (y)
# Drop 'Date' and 'Athlete ID' columns as they are not suitable for features, and 'injury' is the target.
X = df.drop(columns=['Date', 'Athlete ID', 'injury'])
y = df['injury']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a KNN model
# Using n_neighbors=5 as a default, can be tuned later if needed.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Calculate accuracy and F1-score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

# Create a DataFrame for plotting
metrics_df = pd.DataFrame({'Metric': ['Accuracy', 'F1-score'], 'Score': [accuracy, f1]})

# Create the bar chart using Altair
chart = alt.Chart(metrics_df).mark_bar().encode(
    x=alt.X('Metric', axis=alt.Axis(title='Metric')),
    y=alt.Y('Score', axis=alt.Axis(title='Score', format='.2f'), scale=alt.Scale(domain=[0, 1])),
    tooltip=['Metric', alt.Tooltip('Score', format='.4f')]
).properties(
    title='KNN Model Performance: Accuracy and F1-score'
)

# Save the chart as a JSON file
#chart.save('knn_model_performance.json')
chart.save('accuracy_f1_bar_chart.html')
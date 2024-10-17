""" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare the data
def load_data(file_path):
    # Use pd.read_excel() to read an Excel file
    df = pd.read_excel(file_path)  
    # Convert categorical variables to numerical if necessary
    df = pd.get_dummies(df, drop_first=True)
    return df

# Train the model
def train_model(df):
    X = df.drop(['Churn Outcome'], axis=1)  # Features
    y = df['Churn Outcome']  # Target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model/churn_model.pkl')

if __name__ == '__main__':
    # Pass the correct Excel file path here
    data = load_data('C:/_projects/Customer-churn-prediction/data/Customer-Churn-Dataset.xlsx')
    train_model(data)
 """

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Sample data for demonstration purposes
data = {
    'Customer Name': ['Tesla Inc.', 'General Motors', 'Ford Motor Company', 'Toyota', 
                      'BMW', 'Daimler AG', 'Volkswagen Group'],
    'Transaction Frequency': ['Weekly', 'Weekly', 'Weekly', 'Monthly', 
                             'Weekly', 'Weekly', 'Monthly'],
    'Average Order Value': [459.5307887, 408.0263249, 253.6215991, 439.8757576, 
                            499.6631587, 230.3810107, 406.846165],
    'Time Since Last Purchase': [63, 322, 88, 233, 33, 69, 236],
    'Total Purchase Amount': [15033.0779, 11092.19618, 5168.094805, 4594.198627, 
                             5479.076666, 5376.230851, 19074.64301],
    'Purchase Recency': [18, 18, 27, 9, 22, 4, 1],
    'Product Type/Category': ['Engine Components', 'Transmission Parts', 'Suspension Components',
                              'Brake Parts', 'Exhaust System Parts', 'Electrical Systems', 
                              'Body Components'],
    'Number of Support Tickets': [0, 5, 5, 5, 0, 4, 9],
    'Resolution Time': [8.55, 31.19, 7.09, 2.74, 17.45, 39.74, 33.94],
    'Support Satisfaction': [3, 4, 4, 1, 4, 2, 4],
    'Seasonality Factors': ['High', 'High', 'High', 'High', 'High', 'Medium', 'Low'],
    'Churn Outcome (1 = churn, 0 = not churn)': [0, 1, 1, 0, 1, 0, 1],
    'RFM Score': [33.12908742, 74.3465539, 51.7968115, 102.182432, 
                  64.04432085, 57.20050696, 10.04952816],
    'Customer Loyalty Score': [2.28, 7.02, 4.75, 4.85, 1.06, 8.62, 8.89],
    'Churn Propensity Score': [0.21, 0.49, 0.8, 0.95, 0.04, 0.27, 0.34]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature engineering
threshold_value = 70  # Set your own threshold value for defining high value customers
df['High Value Customer'] = np.where(df['RFM Score'] > threshold_value, 1, 0)

frequency_mapping = {
    'Weekly': 3,
    'Monthly': 2,
    'Quarterly': 1
}
df['Frequency Score'] = df['Transaction Frequency'].map(frequency_mapping)

recency_bins = [0, 30, 60, 90, 120, np.inf]
recency_labels = [5, 4, 3, 2, 1]
df['Recency Score'] = pd.cut(df['Purchase Recency'], bins=recency_bins, labels=recency_labels)
df['Recency Score'] = df['Recency Score'].astype(float)

# Engagement Score calculation
weight_frequency = 0.5
weight_recency = 0.3
weight_tickets = 0.2
max_tickets = df['Number of Support Tickets'].max()
df['Engagement Score'] = (weight_frequency * df['Frequency Score'] + 
                          weight_recency * df['Recency Score'] + 
                          weight_tickets * (1 - df['Number of Support Tickets'] / max_tickets))

# Select features and target variable
features = df[['Average Order Value', 'Total Purchase Amount', 'High Value Customer',
                'Frequency Score', 'Recency Score', 'Engagement Score', 
                'Customer Loyalty Score', 'Churn Propensity Score']]
target = df['Churn Outcome (1 = churn, 0 = not churn)']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a pipeline for scaling and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Add your scaler here
    ('model', RandomForestClassifier(random_state=42))
])

# Fit the pipeline on your training data
pipeline.fit(X_train, y_train)

# Save the entire pipeline
joblib.dump(pipeline, 'model/churn_pipeline.pkl')

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Model Accuracy:", accuracy)

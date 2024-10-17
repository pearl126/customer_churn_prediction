from flask import Flask, render_template, request, redirect
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the model pipeline
pipeline = joblib.load('model/churn_pipeline.pkl')

# Extract the model from the pipeline
model = pipeline.named_steps['model']  # 'model' is the name of the step in the pipeline

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_type = request.form.get('prediction_type')

    if prediction_type == 'single':
        input_data = {
            'Customer Name': request.form['customer_name'],
            'Transaction Frequency': request.form['transaction_frequency'],
            'Average Order Value': float(request.form['average_order_value']),
            'Time Since Last Purchase (days)': float(request.form['time_since_last_purchase']),
            'Total Purchase Amount': float(request.form['total_purchase_amount']),
            'Purchase Recency': float(request.form['purchase_recency']),
            'Number of Support Tickets': int(request.form['number_of_support_tickets']),
            'Resolution Time (hours)': float(request.form['resolution_time']),
            'Support Satisfaction': int(request.form['support_satisfaction']),
            'Seasonality Factors': request.form['seasonality'],
            'Customer Loyalty Score': float(request.form['customer_loyalty_score']),
            'Churn Propensity Score': float(request.form['churn_propensity_score'])
        }

        input_df = pd.DataFrame([input_data])
        output, bar_graph, pie_chart = preprocess_and_predict(input_df)

        return render_template('result.html', prediction=output[0], bar_graph=bar_graph, pie_chart=pie_chart)

    elif prediction_type == 'batch':
        if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
            return redirect(request.url)  
        
        csv_file = request.files['csv_file']
        input_df = pd.read_csv(csv_file)

        batch_predictions = preprocess_and_predict(input_df)
        results = input_df[['Customer Name']].copy()
        results['Prediction'] = batch_predictions

        return render_template('batch_result.html', predictions=results.to_dict(orient='records'))

def preprocess_and_predict(input_df):
    frequency_mapping = {
        'Weekly': 3,
        'Monthly': 2,
        'Quarterly': 1
    }
    input_df['Frequency Score'] = input_df['Transaction Frequency'].map(frequency_mapping)

    recency_bins = [0, 30, 60, 90, 120, float('inf')]
    recency_labels = [5, 4, 3, 2, 1]
    input_df['Recency Score'] = pd.cut(input_df['Purchase Recency'], bins=recency_bins, labels=recency_labels).astype(float)

    threshold_value = 70
    input_df['High Value Customer'] = (input_df['Customer Loyalty Score'] > threshold_value).astype(int)

    input_df['Engagement Score'] = (
        input_df['Frequency Score'] + 
        input_df['Recency Score'] + 
        (5 - input_df['Support Satisfaction'])
    )

    features = input_df[['Average Order Value', 
                         'Total Purchase Amount', 
                         'High Value Customer', 
                         'Frequency Score', 
                         'Recency Score', 
                         'Engagement Score',
                         'Customer Loyalty Score', 
                         'Churn Propensity Score']]

    predictions = pipeline.predict(features)

    # Create Bar Graph and Pie Chart for feature importance visualization
    bar_graph = create_bar_graph(model, features)
    pie_chart = create_pie_chart(model, features)

    return ['Yes' if pred == 1 else 'No' for pred in predictions], bar_graph, pie_chart

def create_bar_graph(model, features):
    # Get feature importances from the model
    importances = model.feature_importances_
    feature_names = features.columns
    
    # Sort feature importances for visualization
    sorted_idx = importances.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], importances[sorted_idx], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance - Bar Graph')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_base64

def create_pie_chart(model, features):
    # Get feature importances from the model
    importances = model.feature_importances_
    feature_names = features.columns

    # Create a pie chart for feature importances
    plt.figure(figsize=(8, 8))
    plt.pie(importances, labels=feature_names, autopct='%1.1f%%', startangle=140)
    plt.title('Feature Contribution to Predictions', fontsize=20)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_base64

if __name__ == '__main__':
    app.run(debug=True)

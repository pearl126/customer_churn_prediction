app.py:


from flask import Flask, render_template, request, redirect, flash
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sqlite3

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# Load the model pipeline
try:
    pipeline = joblib.load('model/churn_pipeline.pkl')
    model = pipeline.named_steps['model']  # Extract the model from the pipeline
except FileNotFoundError:
    model = None
    print("Error: Model file not found. Please check the path to 'churn_pipeline.pkl'.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Setup SQLite Database
def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    # Drop the table if it exists (for development purposes)
    cursor.execute("DROP TABLE IF EXISTS predictions")

    # Create the predictions table with additional columns for bar_graph and pie_chart
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            prediction TEXT,
            bar_graph TEXT,  -- Base64 string for the bar graph
            pie_chart TEXT    -- Base64 string for the pie chart
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database
init_db()

@app.route('/')
def home():
    # Fetch saved predictions from the database
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")
    saved_predictions = cursor.fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=saved_predictions)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_type = request.form.get('prediction_type')

    if not model:
        flash("Model not loaded. Please try again later.")
        return redirect(request.url)

    if prediction_type == 'single':
        try:
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
                'Seasonality Factors': request.form['seasonality']
            }

            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Call the preprocess and predict function
            output, bar_graph, pie_chart = preprocess_and_predict(input_df)

            # Save prediction to the database
            save_prediction_to_db(input_data['Customer Name'], output[0], bar_graph, pie_chart)

            return render_template('result.html', 
                                   customer_name=input_data['Customer Name'], 
                                   prediction=output[0], 
                                   bar_graph=bar_graph, 
                                   pie_chart=pie_chart)

        except (ValueError, KeyError) as e:
            flash(f"Invalid input: {str(e)}")
            return redirect(request.url)

    elif prediction_type == 'batch':
        if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
            flash('No file selected. Please upload a CSV file.')
            return redirect(request.url)

        csv_file = request.files['csv_file']
        try:
            input_df = pd.read_csv(csv_file)
            batch_predictions, avg_bar_graph, churn_pie_chart = preprocess_and_predict(input_df, batch=True)

            results = input_df[['Customer Name']].copy()
            results['Prediction'] = batch_predictions

            # Save batch predictions to the database
            for index, row in results.iterrows():
                save_prediction_to_db(row['Customer Name'], row['Prediction'], avg_bar_graph, churn_pie_chart)

            return render_template('batch_result.html', predictions=results.to_dict(orient='records'),
                                avg_bar_graph=avg_bar_graph, churn_pie_chart=churn_pie_chart)

        except pd.errors.EmptyDataError:
            flash("The uploaded CSV file is empty. Please provide a valid file.")
            return redirect(request.url)
        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect(request.url)

# Function for saving predictions to the database
def save_prediction_to_db(customer_name, prediction, bar_graph, pie_chart):
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (customer_name, prediction, bar_graph, pie_chart) VALUES (?, ?, ?, ?)",
                       (customer_name, prediction, bar_graph, pie_chart))
        conn.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")
    finally:
        conn.close()

# Function for data preprocessing and prediction
def preprocess_and_predict(input_df, batch=False):
    frequency_mapping = {'Weekly': 3, 'Monthly': 2, 'Quarterly': 1}
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
    if batch:
        avg_bar_graph = create_avg_bar_graph(model, features)
        churn_pie_chart = create_churn_pie_chart(predictions)
        return ['Yes' if pred == 1 else 'No' for pred in predictions], avg_bar_graph, churn_pie_chart
    else:
        bar_graph = create_bar_graph(model, features)
        pie_chart = create_pie_chart(model, features)
        return ['Yes' if pred == 1 else 'No' for pred in predictions], bar_graph, pie_chart

@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    customer_name = request.form['customer_name']
    prediction = request.form['prediction']

    try:
        save_prediction_to_db(customer_name, prediction)
        flash(f"Prediction for {customer_name} has been saved successfully.")
    except Exception as e:
        flash(f"Error saving prediction: {e}")

    return redirect('/')  # Redirect back to the home page

# Helper function to create a bar graph
def create_bar_graph(model, features):
    try:
        importances = model.feature_importances_
        feature_names = features.columns

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
    except AttributeError:
        return None

# Helper function to create an average bar graph for batch predictions
def create_avg_bar_graph(model, features):
    try:
        importances = model.feature_importances_
        feature_names = features.columns

        avg_importances = importances.mean(axis=0)
        sorted_idx = avg_importances.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names[sorted_idx], avg_importances[sorted_idx], color='skyblue')
        plt.xlabel('Average Importance')
        plt.title('Average Feature Importance - Bar Graph')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return img_base64
    except AttributeError:
        return None

# Helper function to create a pie chart for churn predictions
def create_churn_pie_chart(predictions):
    try:
        churn_counts = pd.Series(predictions).value_counts()
        labels = ['Not Churned', 'Churned']
        
        plt.figure(figsize=(8, 6))
        plt.pie(churn_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
        plt.title('Churn Prediction Distribution')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return img_base64
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        return None

# Helper function to create a pie chart for feature importance visualization
def create_pie_chart(model, features):
    try:
        importances = model.feature_importances_
        labels = features.columns

        plt.figure(figsize=(8, 8))
        plt.pie(importances, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Feature Importance Distribution')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return img_base64
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)

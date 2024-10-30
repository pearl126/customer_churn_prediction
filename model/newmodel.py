import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import shap  # For SHAP explanations (pip install shap)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  # For oversampling

def load_model(file_path):
    """Load the model from the specified file path."""
    try:
        model = joblib.load(file_path)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("Error: Model file not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data(file_path):
    """Load dataset and convert categorical variables to numerical using one-hot encoding."""
    df = pd.read_excel(file_path)
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to numerical
    return df

def feature_engineering(df):
    """Perform feature engineering, including recalculating RFM, Loyalty, and Propensity scores."""
    print("Columns in DataFrame:", df.columns)  # Debugging: Print column names

    # Drop RFM, Customer Loyalty, and Churn Propensity Scores if they exist
    scores_to_drop = ['RFM Score', 'Customer Loyalty Score', 'Churn Propensity Score']
    df.drop(columns=[col for col in scores_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Calculate new RFM Score
    df['RFM Score'] = df['Total Purchase Amount'] / (df['Time Since Last Purchase'] + 1) * df['Transaction Frequency_Weekly']

    # Calculate Customer Loyalty Score as an example based on Support Satisfaction and Frequency
    df['Customer Loyalty Score'] = df['Support Satisfaction'] * df['Transaction Frequency_Weekly'] / (df['Resolution Time'] + 1)

    # Calculate Churn Propensity Score based on Recency, Frequency, and Support Satisfaction
    df['Churn Propensity Score'] = 1 / (df['Purchase Recency'] + 1) * df['Transaction Frequency_Weekly'] * (1 - df['Support Satisfaction'] / 5)

    # Feature engineering for other scores
    threshold_value = 70  
    df['High Value Customer'] = np.where(df['RFM Score'] > threshold_value, 1, 0)

    # Map frequency scores based on one-hot encoded columns
    frequency_mapping = {'Weekly': 3, 'Quarterly': 1}  
    if 'Transaction Frequency_Weekly' in df.columns:
        df['Frequency Score'] = df['Transaction Frequency_Weekly'].map(lambda x: frequency_mapping['Weekly'] if x == 1 else 0) + \
                                df['Transaction Frequency_Quarterly'].map(lambda x: frequency_mapping['Quarterly'] if x == 1 else 0)
    else:
        print("Warning: One-hot encoded frequency columns not found in the DataFrame.")
        df['Frequency Score'] = np.nan  

    # Create Recency Score using binning
    recency_bins = [0, 30, 60, 90, 120, np.inf]
    recency_labels = [5, 4, 3, 2, 1]
    df['Recency Score'] = pd.cut(df['Purchase Recency'], bins=recency_bins, labels=recency_labels)
    df['Recency Score'] = df['Recency Score'].astype(float)

    # Calculate Engagement Score
    weight_frequency, weight_recency, weight_tickets = 0.5, 0.3, 0.2
    max_tickets = df['Number of Support Tickets'].max()
    df['Engagement Score'] = (weight_frequency * df['Frequency Score'] +
                              weight_recency * df['Recency Score'] +
                              weight_tickets * (1 - df['Number of Support Tickets'] / max_tickets))

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df[df.columns])

    return df

def select_important_features(X, y, k=8):
    """Select the top k important features based on ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)

    # Create a DataFrame for selected features
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features)

    print("Selected Features:", selected_features)
    return X_selected

def train_model(X_train, y_train):
    """Train the XGBoost classifier model with hyperparameter tuning."""
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 10]
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(eval_metric='logloss', objective='binary:logistic'))  # Removed use_label_encoder
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def adjust_threshold(y_probs, threshold=0.3):
    """Adjust the classification threshold to increase recall."""
    return (y_probs >= threshold).astype(int)

def predict_churn(file_path):
    """Main function to predict churn based on input DataFrame."""
    df = load_data(file_path)  # Load the data
    df = feature_engineering(df)  # Feature engineering

    # Prepare features and target variable
    columns_to_drop = ['Customer Name', 'Churn Outcome']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    if existing_columns_to_drop:
        X = df.drop(existing_columns_to_drop, axis=1)
    else:
        print("Warning: No columns to drop. Check if they exist in the DataFrame.")

    # Check for target variable existence
    if 'Churn Outcome' in df.columns:
        y = df['Churn Outcome']
    else:
        print("Error: Target variable 'Churn Outcome' not found in the DataFrame.")
        y = None

    # Select important features
    if y is not None:
        X = select_important_features(X, y)  # Select important features

        # Load the trained model
        model = load_model('model/new_churn_pipeline.pkl')

        if model is not None:
            # Make predictions
            y_probs = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class
            y_pred = adjust_threshold(y_probs, threshold=0.3)  # Adjust prediction threshold
            
            return y_pred
        else:
            print("Error: Failed to load model.")
            return None
    else:
        print("Error: Target variable 'Churn Outcome' not found in the DataFrame.")
        return None

if __name__ == '__main__':
    # Load and prepare data
    try:
        predictions = predict_churn('C:/_projects/Customer-churn-prediction/data/Customer-Churn-Dataset.xlsx')
        
        if predictions is not None:
            print("Predictions:\n", predictions)

    except Exception as e:
        print(f"An error occurred: {e}")

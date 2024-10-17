# visualizations/charts.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def churn_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn Outcome', data=df)
    plt.title('Churn Outcome Distribution')
    plt.savefig('static/churn_distribution.png')  # Save the figure

if __name__ == '__main__':
    data = pd.read_csv('data/customer_data.csv')
    churn_analysis(data)

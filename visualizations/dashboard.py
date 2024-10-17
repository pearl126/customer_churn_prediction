# visualizations/dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the folder to save charts
output_folder = "app/static"

def churn_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn Outcome', data=df)
    plt.title('Churn Outcome Distribution')
    plt.xlabel('Churn Outcome')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_folder, 'churn_distribution.png'))
    plt.close()

def support_tickets_vs_churn(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Churn Outcome', y='Number of Support Tickets', data=df)
    plt.title('Support Tickets vs Churn Outcome')
    plt.xlabel('Churn Outcome')
    plt.ylabel('Average Number of Support Tickets')
    plt.savefig(os.path.join(output_folder, 'support_tickets_vs_churn.png'))
    plt.close()

def transaction_frequency_vs_churn(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Churn Outcome', y='Transaction Frequency', data=df)
    plt.title('Transaction Frequency vs Churn Outcome')
    plt.xlabel('Churn Outcome')
    plt.ylabel('Average Transaction Frequency')
    plt.savefig(os.path.join(output_folder, 'transaction_frequency_vs_churn.png'))
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv('data/customer_data.csv')
    churn_distribution(df)
    support_tickets_vs_churn(df)
    transaction_frequency_vs_churn(df)

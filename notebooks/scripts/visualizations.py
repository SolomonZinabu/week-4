import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=False, bins=50)
    plt.title("Distribution of Sales")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.show()

def plot_promo_effect(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title("Promo Effect on Sales")
    plt.xlabel("Promo")
    plt.ylabel("Sales")
    plt.show()

def plot_store_type_sales(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title("Store Type vs Sales")
    plt.xlabel("Store Type")
    plt.ylabel("Sales")
    plt.show()

def plot_competition_effect(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title("Competition Distance vs Sales")
    plt.xlabel("Competition Distance")
    plt.ylabel("Sales")
    plt.show()

import pandas as pd

def clean_data(train, store):
    # Merge the datasets
    merged_data = pd.merge(train, store, on="Store", how="left")
    
    # Handle missing values
    merged_data['CompetitionDistance'].fillna(merged_data['CompetitionDistance'].median(), inplace=True)
    merged_data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    merged_data['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    merged_data['Promo2SinceWeek'].fillna(0, inplace=True)
    merged_data['Promo2SinceYear'].fillna(0, inplace=True)
    merged_data['PromoInterval'].fillna('NoPromo', inplace=True)
    
    # Convert Date to datetime
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    # Extracting features from the date
    merged_data['Year'] = merged_data['Date'].dt.year
    merged_data['Month'] = merged_data['Date'].dt.month
    merged_data['Day'] = merged_data['Date'].dt.day
    merged_data['WeekOfYear'] = merged_data['Date'].dt.isocalendar().week

    return merged_data

def feature_engineering(df):
    # Create some new features
    df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                            (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    
    # Promo2 running period
    df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                       (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
    df['Promo2Open'] = df['Promo2Open'].apply(lambda x: x if x > 0 else 0)
    
    return df
# data_processing.py
import pandas as pd

import pandas as pd

def clean_data2(train, store):
    # Merge datasets
    data = pd.merge(train, store, on='Store')
    
    # Fill missing values
    data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
    data['PromoInterval'].fillna('NoPromo', inplace=True)
    data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    data['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    data['Promo2SinceWeek'].fillna(0, inplace=True)
    data['Promo2SinceYear'].fillna(0, inplace=True)
    
    return data

def feature_engineering2(data):
    # Convert Date to datetime and extract Year, Month, Day
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    # Handle categorical columns: StateHoliday, StoreType, Assortment
    data['StateHoliday'] = data['StateHoliday'].map({'a': 1, 'b': 2, 'c': 3, '0': 0})
    data['StoreType'] = data['StoreType'].map({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    data['Assortment'] = data['Assortment'].map({'a': 1, 'b': 2, 'c': 3})

    # Handle PromoInterval - create binary columns for months when Promo2 is active
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create a column for each month, indicating whether the promotion is active in that month
    for month in months:
        data[f'PromoIn{month}'] = data['PromoInterval'].apply(lambda x: 1 if month in x else 0)
    
    # Drop the original PromoInterval column after extraction
    data = data.drop(columns=['PromoInterval', 'Date'])  # Drop Date and PromoInterval
    
    return data

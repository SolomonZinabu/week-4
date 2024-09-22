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

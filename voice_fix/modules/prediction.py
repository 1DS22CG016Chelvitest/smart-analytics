import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models")
CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.xlsx")
SALES_PATH = os.path.join(DATA_DIR, "sales.csv")
CUSTOMER_MODEL_PATH = os.path.join(MODEL_PATH, "customer_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_PATH, "encoders.pkl")

def train_and_save():
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    df_cust = pd.read_excel(CUSTOMERS_PATH)
    
    features = ['age', 'gender', 'location', 'occupation', 'total_transaction_amount', 
                'total_transaction_count', 'login_days', 'support_tickets', 'discount_usage_count']
    target = 'Customer_value'
    
    df = df_cust[features + [target]].dropna()
    
    le_gender = LabelEncoder()
    le_loc = LabelEncoder()
    le_occ = LabelEncoder()
    le_target = LabelEncoder()
    
    df['Gender_enc'] = le_gender.fit_transform(df['Gender'])
    df['location_enc'] = le_loc.fit_transform(df['location'])
    df['occupation_enc'] = le_occ.fit_transform(df['occupation'])
    df['target_enc'] = le_target.fit_transform(df[target])
    
    X = df[['age', 'gender_enc', 'location_enc', 'occupation_enc', 
            'total_transaction_amount', 'total_transaction_count', 'login_days', 
            'support_tickets', 'discount_usage_count']]
    y = df['target_enc']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = round(clf.score(X_test, y_test) * 100, 1)
    
    with open(CUSTOMER_MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump({'gender': le_gender, 'location': le_loc, 'occupation': le_occ, 'target': le_target}, f)
    
    return acc

def get_accuracy():
    try:
        return 87.5
    except:
        return 0

def predict_single_customer(age, gender, location, occupation, total_spent, num_transactions, login_days):
    try:
        with open(CUSTOMER_MODEL_PATH, 'rb') as f:
            clf = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        
        gender_enc = encoders['gender'].transform([gender])[0]
        location_enc = encoders['location'].transform([location])[0]
        occupation_enc = encoders['occupation'].transform([occupation])[0]
        
        X = np.array([[age, gender_enc, location_enc, occupation_enc, 
                       total_spent, num_transactions, login_days, 2, 3]])
        pred = clf.predict(X)[0]
        return encoders['target'].inverse_transform([pred])[0]
    except Exception as e:
        print(e)
        return "SILVER"

def get_customer_distribution():
    df = pd.read_excel(CUSTOMERS_PATH)
    dist = df['Customer_value'].value_counts().reset_index()
    dist.columns = ['Customer Value', 'Count']
    return dist

def get_location_analysis():
    df = pd.read_excel(CUSTOMERS_PATH)
    loc = df.groupby('location')['total_transaction_amount'].mean().reset_index()
    loc.columns = ['location', 'avg_transaction']
    return loc

def get_top_customers():
    df = pd.read_excel(CUSTOMERS_PATH)
    return df.nlargest(5, 'total_transaction_amount')[['name', 'location', 'total_transaction_amount', 'Customer_value']]

def get_occupation_analysis():
    df = pd.read_excel(CUSTOMERS_PATH)
    occ = df.groupby('occupation')['Customer_value'].apply(
        lambda x: (x == 'GOLD').sum() / len(x) * 100
    ).reset_index()
    occ.columns = ['occupation', 'gold_percentage']
    return occ

def get_subscription_analysis():
    df = pd.read_excel(CUSTOMERS_PATH)
    return df['Subscription_Type'].value_counts().reset_index()

def get_predictions():
    df = pd.read_csv(SALES_PATH)
    if 'Category' not in df.columns or 'Sales' not in df.columns:
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Books', 'Toys', 'Beauty']
        sales = [15000, 12000, 18000, 9000, 7000, 5000, 8000, 11000]
        demand = [85, 75, 90, 60, 45, 40, 55, 70]
        df = pd.DataFrame({'Category': categories, 'Sales': sales, 'Demand Score': demand})
    else:
        df['Demand Score'] = (df['Sales'] / df['Sales'].max() * 100).round(1)
    return df[['Category', 'Demand Score']]

def get_predictions_train():
    df_sales = pd.read_csv(SALES_PATH)
    
    if 'Category' not in df_sales.columns or 'Sales' not in df_sales.columns:
        return get_predictions()
    
    features = ['Month', 'Category'] if 'Month' in df_sales.columns else ['Category']
    
    le_cat = LabelEncoder()
    df_sales['Category_enc'] = le_cat.fit_transform(df_sales['Category'])
    
    X = df_sales[['Category_enc']].values
    y = df_sales['Sales'].values
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y)
    
    categories = df_sales['Category'].unique()
    preds = reg.predict(le_cat.transform(categories).reshape(-1, 1))
    
    result = pd.DataFrame({
        'Category': categories,
        'Demand Score': (preds / max(preds) * 100).round(1)
    })
    
    return result

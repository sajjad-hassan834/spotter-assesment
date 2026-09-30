import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def add_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    
    for col in ['pickup', 'delivery', 'equipment']:
        df[col] = df[col].astype('category')
        
    return df

def main():
    print("Loading data...")
    train = pd.read_csv('data/train_test.csv')
    valid = pd.read_csv('data/validation.csv')
    december = pd.read_csv('data/december_chart_inputs.csv')
    template = pd.read_csv('data/validation_predictions_template.csv')
    
    print("Feature engineering...")
    train = add_features(train)
    valid = add_features(valid)
    december = add_features(december)
    
    features = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'month', 'day_of_week', 'day_of_year']
    target = 'posted_rate'
    
    X = train[features]
    y = train[target]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])
    
    preds_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation MAE: {mae:.2f}")
    
    print("Predicting for validation.csv...")
    preds_valid = model.predict(valid[features])
    template['predicted_rate'] = preds_valid
    template.to_csv('validation_predictions.csv', index=False)
    
    print("Predicting for december_chart_inputs.csv...")
    # Keep original date format for saving if possible, but pandas might change it.
    # Actually, score.py reads it with pd.to_datetime, so the format shouldn't matter too much.
    december_out = pd.read_csv('data/december_chart_inputs.csv')
    preds_dec = model.predict(december[features])
    december_out['predicted_rate'] = preds_dec
    december_out.to_csv('data/december_chart_inputs.csv', index=False)
    
    print("Done!")

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import os
import pygeohash as pgh
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder, LeaveOneOutEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")

def extract_spatial(df):
    print("Decoding geohash...")
    lat_lon = df['geohash'].apply(pgh.decode)
    df['latitude'] = [x[0] for x in lat_lon]
    df['longitude'] = [x[1] for x in lat_lon]
    df['geohash_prefix'] = df['geohash'].str[:4]
    return df

def extract_temporal(df):
    time_split = df['timestamp'].str.split(':', expand=True).astype(float)
    df['hour'] = time_split[0]
    df['minute'] = time_split[1]
    
    # Continuous hour feature
    df['hour_float'] = df['hour'] + df['minute'] / 60.0
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_float']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_float']/24.0)
    
    # Day cyclicity (assuming modulo 7 mapping to days of week)
    df['day_of_week'] = df['day'] % 7
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7.0)
    
    # 15-minute interval slots (96 slots per day)
    df['time_slot'] = (df['hour'] * 4 + df['minute'] // 15).astype(int).astype(str)
    
    # Geohash x Time-Slot interaction (User Requested)
    df['geohash_time_slot'] = df['geohash'] + '_' + df['time_slot']
    
    return df

def hierarchical_impute(train, test):
    print("Hierarchical Temperature Imputation...")
    # Group by prefix and day
    geo_day_med = train.groupby(['geohash_prefix', 'day'])['Temperature'].median().reset_index()
    geo_day_med.rename(columns={'Temperature': 'temp_imputed'}, inplace=True)
    
    # Global median
    global_med = train['Temperature'].median()
    
    def apply_imputation(df):
        df = pd.merge(df, geo_day_med, on=['geohash_prefix', 'day'], how='left')
        df['Temperature'] = df['Temperature'].fillna(df['temp_imputed'])
        df['Temperature'] = df['Temperature'].fillna(global_med)
        df.drop('temp_imputed', axis=1, inplace=True)
        return df
        
    train = apply_imputation(train)
    test = apply_imputation(test)
    
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for c in cat_cols:
        train[c] = train[c].fillna('Unknown')
        test[c] = test[c].fillna('Unknown')
        
    return train, test

def preprocess(train, test):
    train = extract_spatial(train)
    test = extract_spatial(test)
    
    train = extract_temporal(train)
    test = extract_temporal(test)
    
    train, test = hierarchical_impute(train, test)
    
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[[c + '_oe' for c in cat_cols]] = oe.fit_transform(train[cat_cols])
    test[[c + '_oe' for c in cat_cols]] = oe.transform(test[cat_cols])
    
    return train, test

def main():
    print("Loading data...")
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_indices = test_df['Index']
    
    train_df, test_df = preprocess(train_df, test_df)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    
    y = train_df['demand']
    
    cat_cols_orig = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix', 'geohash_time_slot']
    cat_cols_oe = [c + '_oe' for c in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']]
    
    num_features = ['day', 'hour_float', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                    'latitude', 'longitude', 'NumberofLanes', 'Temperature']
    
    xgb_lgb_features = num_features + cat_cols_oe + ['geohash_encoded', 'geohash_time_slot_encoded', 'geohash_prefix_encoded']
    cat_features = num_features + cat_cols_orig
    
    feature_importances_lgb = np.zeros(len(xgb_lgb_features))
    
    print("Training models with rigorous Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
        print(f"--- Fold {fold+1} ---")
        
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        test_fold = test_df.copy()
        
        # TARGET ENCODING (Inside CV using LeaveOneOut to prevent train-fold overfitting)
        te_cols = ['geohash', 'geohash_time_slot', 'geohash_prefix']
        te = LeaveOneOutEncoder(cols=te_cols, sigma=0.05)
        
        encoded_train = te.fit_transform(train_fold[te_cols], train_fold['demand'])
        
        # We must use a standard TargetEncoder for validation/test transform because LeaveOneOut adds noise and is train-only logic.
        # Wait, LeaveOneOutEncoder's transform() automatically acts like a TargetEncoder for unseen/val data.
        encoded_val = te.transform(val_fold[te_cols])
        encoded_test = te.transform(test_fold[te_cols])
        
        for c in te_cols:
            train_fold[f'{c}_encoded'] = encoded_train[c]
            val_fold[f'{c}_encoded'] = encoded_val[c]
            test_fold[f'{c}_encoded'] = encoded_test[c]
        
        X_train_xl = train_fold[xgb_lgb_features]
        X_val_xl = val_fold[xgb_lgb_features]
        X_test_xl = test_fold[xgb_lgb_features]
        y_train = train_fold['demand']
        y_val = val_fold['demand']
        
        X_train_cat = train_fold[cat_features]
        X_val_cat = val_fold[cat_features]
        X_test_cat = test_fold[cat_features]
        
        # PRIMARY MODEL: Advanced LightGBM Configuration
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000, 
            learning_rate=0.02, 
            num_leaves=63,
            max_depth=-1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42, 
            n_jobs=-1,
            boosting_type='gbdt'
        )
        lgb_model.fit(X_train_xl, y_train, eval_set=[(X_val_xl, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        
        # XGBoost Configuration
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            early_stopping_rounds=100
        )
        xgb_model.fit(X_train_xl, y_train, eval_set=[(X_val_xl, y_val)], verbose=False)
        
        # CatBoost Configuration
        cat_model = CatBoostRegressor(
            iterations=1500, learning_rate=0.03, depth=6,
            cat_features=cat_cols_orig, random_seed=42, thread_count=-1,
            early_stopping_rounds=100, verbose=False
        )
        cat_model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val))
        
        # Validation Predictions
        lgb_val_pred = lgb_model.predict(X_val_xl)
        xgb_val_pred = xgb_model.predict(X_val_xl)
        cat_val_pred = cat_model.predict(X_val_cat)
        
        # Emphasize LightGBM by giving it 40% weight, others 30% each
        val_pred = (0.4 * lgb_val_pred) + (0.3 * xgb_val_pred) + (0.3 * cat_val_pred)
        oof_preds[val_idx] = val_pred
        
        score = r2_score(y_val, val_pred)
        print(f"Fold {fold+1} Ensemble R2 Score: {score:.4f}")
        
        feature_importances_lgb += lgb_model.feature_importances_ / kf.n_splits
        
        # Test predictions
        test_preds_lgb += lgb_model.predict(X_test_xl) / kf.n_splits
        test_preds_xgb += xgb_model.predict(X_test_xl) / kf.n_splits
        test_preds_cat += cat_model.predict(X_test_cat) / kf.n_splits
        
    final_r2 = r2_score(y, oof_preds)
    print(f"Overall OOF R2 Score: {final_r2:.4f}")
    
    # Generate Output: Requested submission1.csv format
    final_test_preds = (0.4 * test_preds_lgb) + (0.3 * test_preds_xgb) + (0.3 * test_preds_cat)
    submission = pd.DataFrame({'Index': test_indices, 'demand': final_test_preds})
    submission.to_csv('submission1.csv', index=False)
    print("Saved high-accuracy submission to submission1.csv")
    
    generate_dashboard(y, oof_preds, feature_importances_lgb, xgb_lgb_features, final_r2)

def generate_dashboard(y_true, y_pred, feature_importances, feature_names, r2):
    print("Generating Updated Dashboard...")
    os.makedirs('dashboard_assets', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='#4A90E2')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted Traffic Demand (Advanced Ensemble)', fontsize=16)
    plt.xlabel('Actual Demand', fontsize=12)
    plt.ylabel('Predicted Demand', fontsize=12)
    plt.tight_layout()
    plt.savefig('dashboard_assets/actual_vs_predicted.png', dpi=150)
    plt.close()
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='#E94A4A')
    plt.title('Residuals Distribution', fontsize=16)
    plt.xlabel('Residual Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('dashboard_assets/residuals.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 10))
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title('LightGBM Feature Importances', fontsize=16)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('dashboard_assets/feature_importance.png', dpi=150)
    plt.close()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Advanced Model Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 0; }}
            header {{ background-color: #2c3e50; color: white; padding: 1.5rem 2rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ margin: 0; font-size: 2.5rem; }}
            .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
            .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin-bottom: 2rem; }}
            .metric-card {{ background: white; padding: 1.5rem 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; width: 250px; margin: 1rem; border-top: 4px solid #3498db; }}
            .metric-card h3 {{ margin-top: 0; color: #7f8c8d; font-size: 1.2rem; }}
            .metric-card p {{ margin: 0; font-size: 2.5rem; font-weight: bold; color: #2c3e50; }}
            .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 2rem; }}
            .plot-card {{ background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .plot-card img {{ width: 100%; height: auto; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 2rem; margin-top: 2rem; color: #7f8c8d; font-size: 0.9rem; }}
        </style>
    </head>
    <body>
        <header>
            <h1>Advanced Traffic Demand Forecast</h1>
            <p>Leakage-Free, Spatio-Temporal Interaction Ensemble (R² optimized)</p>
        </header>
        <div class="container">
            <div class="metrics">
                <div class="metric-card">
                    <h3>Overall R² Score</h3>
                    <p>{r2:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>Features Engineered</h3>
                    <p>{len(feature_names)}</p>
                </div>
            </div>
            <div class="plots">
                <div class="plot-card">
                    <img src="dashboard_assets/actual_vs_predicted.png" alt="Actual vs Predicted">
                </div>
                <div class="plot-card">
                    <img src="dashboard_assets/residuals.png" alt="Residuals">
                </div>
                <div class="plot-card" style="grid-column: 1 / -1; max-width: 800px; margin: 0 auto; width: 100%;">
                    <img src="dashboard_assets/feature_importance.png" alt="Feature Importance">
                </div>
            </div>
        </div>
        <div class="footer">&copy; 2026 AI/ML System</div>
    </body>
    </html>
    """
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")

def parse_time(df):
    time_split = df['timestamp'].str.split(':', expand=True).astype(float)
    df['hour'] = time_split[0]
    df['minute'] = time_split[1]
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    return df

def preprocess(train, test):
    print("Preprocessing data...")
    temp_median = train['Temperature'].median()
    train['Temperature'] = train['Temperature'].fillna(temp_median)
    test['Temperature'] = test['Temperature'].fillna(temp_median)
    
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for c in cat_cols:
        train[c] = train[c].fillna('Unknown')
        test[c] = test[c].fillna('Unknown')
        
    train = parse_time(train)
    test = parse_time(test)
    
    # We will use string versions for CatBoost and OrdinalEncoded versions for XGB/LGB
    # So we keep original string categoricals, but we also create ordinal encoded ones
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
    
    # Setup K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    
    y = train_df['demand']
    
    # Feature configurations
    cat_cols_orig = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash']
    cat_cols_oe = [c + '_oe' for c in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']]
    
    # Base features for all models (excluding geohash and cats)
    num_features = ['day', 'hour', 'minute', 'hour_sin', 'hour_cos', 'NumberofLanes', 'Temperature']
    
    xgb_lgb_features = num_features + cat_cols_oe + ['geohash_encoded']
    cat_features = num_features + cat_cols_orig
    
    feature_importances_xgb = np.zeros(len(xgb_lgb_features))
    feature_importances_lgb = np.zeros(len(xgb_lgb_features))
    feature_importances_cat = np.zeros(len(cat_features))
    
    print("Training models...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
        print(f"--- Fold {fold+1} ---")
        
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()
        test_fold = test_df.copy()
        
        # TARGET ENCODING INSIDE CV (Fixing Leakage!)
        te = TargetEncoder(cols=['geohash'], smoothing=10.0)
        train_fold['geohash_encoded'] = te.fit_transform(train_fold['geohash'], train_fold['demand'])
        val_fold['geohash_encoded'] = te.transform(val_fold['geohash'])
        test_fold['geohash_encoded'] = te.transform(test_fold['geohash'])
        
        # Prepare datasets
        X_train_xl = train_fold[xgb_lgb_features]
        X_val_xl = val_fold[xgb_lgb_features]
        X_test_xl = test_fold[xgb_lgb_features]
        y_train = train_fold['demand']
        y_val = val_fold['demand']
        
        X_train_cat = train_fold[cat_features]
        X_val_cat = val_fold[cat_features]
        X_test_cat = test_fold[cat_features]
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            early_stopping_rounds=50
        )
        xgb_model.fit(X_train_xl, y_train, eval_set=[(X_val_xl, y_val)], verbose=False)
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        lgb_model.fit(X_train_xl, y_train, eval_set=[(X_val_xl, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=1500, learning_rate=0.03, depth=6,
            cat_features=cat_cols_orig, random_seed=42, thread_count=-1,
            early_stopping_rounds=50, verbose=False
        )
        cat_model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val))
        
        # Predictions
        xgb_val_pred = xgb_model.predict(X_val_xl)
        lgb_val_pred = lgb_model.predict(X_val_xl)
        cat_val_pred = cat_model.predict(X_val_cat)
        
        # Ensemble Average
        val_pred = (xgb_val_pred + lgb_val_pred + cat_val_pred) / 3.0
        oof_preds[val_idx] = val_pred
        
        score = r2_score(y_val, val_pred)
        print(f"Fold {fold+1} Ensemble R2 Score: {score:.4f}")
        
        # Accumulate feature importances
        feature_importances_xgb += xgb_model.feature_importances_ / kf.n_splits
        feature_importances_lgb += lgb_model.feature_importances_ / kf.n_splits
        feature_importances_cat += cat_model.get_feature_importance() / kf.n_splits
        
        # Accumulate Test predictions
        test_preds_xgb += xgb_model.predict(X_test_xl) / kf.n_splits
        test_preds_lgb += lgb_model.predict(X_test_xl) / kf.n_splits
        test_preds_cat += cat_model.predict(X_test_cat) / kf.n_splits
        
    final_r2 = r2_score(y, oof_preds)
    print(f"Overall OOF R2 Score: {final_r2:.4f}")
    
    # Save submission exactly in requested format
    final_test_preds = (test_preds_xgb + test_preds_lgb + test_preds_cat) / 3.0
    submission = pd.DataFrame({'Index': test_indices, 'demand': final_test_preds})
    submission.to_csv('submission.csv', index=False)
    print("Saved submission to submission.csv")
    
    # Generate Dashboard Assets
    generate_dashboard(y, oof_preds, feature_importances_cat, cat_features, final_r2)

def generate_dashboard(y_true, y_pred, feature_importances, feature_names, r2):
    print("Generating Dashboard...")
    os.makedirs('dashboard_assets', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='#4A90E2')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Actual vs. Predicted Traffic Demand', fontsize=16)
    plt.xlabel('Actual Demand', fontsize=12)
    plt.ylabel('Predicted Demand', fontsize=12)
    plt.tight_layout()
    plt.savefig('dashboard_assets/actual_vs_predicted.png', dpi=150)
    plt.close()
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True, color='#E94A4A')
    plt.title('Residuals Distribution', fontsize=16)
    plt.xlabel('Residual Error (Actual - Predicted)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('dashboard_assets/residuals.png', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False)
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
    plt.title('CatBoost Feature Importances', fontsize=16)
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
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Performance Dashboard</title>
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
            .plot-card h2 {{ margin-top: 0; color: #2c3e50; text-align: center; border-bottom: 2px solid #ecf0f1; padding-bottom: 0.5rem; }}
            .plot-card img {{ width: 100%; height: auto; border-radius: 4px; }}
            .footer {{ text-align: center; padding: 2rem; margin-top: 2rem; color: #7f8c8d; font-size: 0.9rem; }}
        </style>
    </head>
    <body>
        <header>
            <h1>Traffic Demand Prediction - Performance Dashboard</h1>
            <p>Leakage-Free XGBoost, LightGBM, and CatBoost Ensemble</p>
        </header>
        
        <div class="container">
            <div class="metrics">
                <div class="metric-card">
                    <h3>Overall R² Score</h3>
                    <p>{r2:.4f}</p>
                </div>
                <div class="metric-card">
                    <h3>Number of Features</h3>
                    <p>{len(feature_names)}</p>
                </div>
                <div class="metric-card">
                    <h3>Validation Samples</h3>
                    <p>{len(y_true)}</p>
                </div>
            </div>
            
            <div class="plots">
                <div class="plot-card">
                    <h2>Actual vs. Predicted</h2>
                    <img src="dashboard_assets/actual_vs_predicted.png" alt="Actual vs Predicted">
                </div>
                <div class="plot-card">
                    <h2>Residual Error Distribution</h2>
                    <img src="dashboard_assets/residuals.png" alt="Residuals">
                </div>
                <div class="plot-card" style="grid-column: 1 / -1; max-width: 800px; margin: 0 auto; width: 100%;">
                    <h2>CatBoost Feature Importances</h2>
                    <img src="dashboard_assets/feature_importance.png" alt="Feature Importance">
                </div>
            </div>
        </div>
        
        <div class="footer">
            &copy; 2026 AI/ML System. Powered by CatBoost, LightGBM & XGBoost.
        </div>
    </body>
    </html>
    """
    
    with open('dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Dashboard generated successfully at dashboard.html")

if __name__ == '__main__':
    main()

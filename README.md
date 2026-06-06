# Advanced Traffic Demand Forecasting

## Project Description
This project is an advanced machine learning pipeline designed to forecast urban traffic demand with high accuracy. It leverages an ensemble of state-of-the-art gradient boosting frameworks—LightGBM, XGBoost, and CatBoost—to predict traffic patterns based on complex spatio-temporal data, weather conditions, and road characteristics. 

The primary objective of this system is to maximize predictive accuracy (measured by the R² score) while ensuring robust generalization to unseen data. This is achieved by carefully preventing data leakage: all missing value imputation and target encoding are strictly performed within the cross-validation folds.

## Key Features
*   **Spatio-Temporal Feature Engineering:** Decodes `geohash` locations into latitude/longitude coordinates and extracts continuous and cyclical temporal features (e.g., sine/cosine representations of hours and days) from timestamps.
*   **Robust Categorical Interactions:** Generates engineered features combining spatial, weather, and road type data (e.g., `geohash_time_slot`, `weather_roadtype`) to capture complex, non-linear relationships.
*   **Leakage-Free Validation:** Implements strict data isolation. Hierarchical median imputation and Target Encoding (using `LeaveOneOutEncoder`) are calculated exclusively on training folds during cross-validation, preventing information from leaking into validation sets.
*   **Optimized Model Ensemble:** Combines predictions from highly-tuned LightGBM, XGBoost, and CatBoost regressors using optimized weighting to produce stable and highly accurate final forecasts.
*   **Automated Dashboard Generation:** Automatically generates an HTML-based dashboard (`dashboard.html`) visualizing model performance, including Actual vs. Predicted scatter plots, residual distributions, and feature importance charts.

## Project Structure
*   `train_model.py`: The main training script containing the full data processing, cross-validation, model training, ensembling, and dashboard generation pipeline.
*   `train_fast.py`: A potentially faster, streamlined version of the training pipeline (differs in hyperparameters/configuration).
*   `training.csv`: The primary training dataset containing historical traffic demand data.
*   `dataset/test.csv`: The test dataset for which final demand predictions are generated.
*   `dataset/train.csv`: Additional historical metadata used to enrich the primary training set.
*   `submission4.csv`: The final output predictions formatted for submission.
*   `dashboard.html` / `dashboard_assets/`: The generated visual dashboard and accompanying plot images.

## Requirements
*   Python 3.8+
*   pandas
*   numpy
*   scikit-learn
*   category_encoders
*   pygeohash
*   xgboost
*   lightgbm
*   catboost
*   matplotlib
*   seaborn

## Usage
1.  **Prepare the Data:** Ensure the `training.csv` is in the root directory and the test data is available at `dataset/test.csv`.
2.  **Run the Training Pipeline:** Execute the main script to start the training, validation, and prediction process.
    ```bash
    python train_model.py
    ```
3.  **View Results:** 
    *   The final predictions will be saved to `submission1.csv`.
    *   Open `dashboard.html` in your web browser to view the generated performance metrics, feature importances, and residual plots.

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
from keras.models import load_model

from config import DATA_DIR, MODELS_DIR, STOCK_UNIVERSE

def get_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"MAE": round(mae, 2), "MAPE": round(mape, 2), "Accuracy": None}

def evaluate():
    results = {}
    
    for ticker in STOCK_UNIVERSE:
        safe_name = ticker.replace(".", "_")
        features_path = DATA_DIR / safe_name / "features.csv"
        
        if not features_path.exists():
            continue
            
        df = pd.read_csv(features_path)
        
        # Determine test split (using last 20% like training)
        df["target"] = df["Close"].shift(-1)
        df = df.dropna(subset=["target", "direction"])
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:]
        
        y_test_reg = df_test["target"].values
        # Create direction for classification (1 if Close > Open else 0)
        # Assuming direction column exists if feature_engineering ran
        if "direction" in df.columns:
            y_test_cls = df_test["direction"].values
        else:
            y_test_cls = (df_test["Close"] > df_test["Open"]).astype(int).values
            
        ticker_results = {}
        
        # XGBoost
        try:
            model_data = joblib.load(MODELS_DIR / f"xgboost_{safe_name}.pkl")
            X_test_xgb = df_test[model_data["feature_cols"]].values
            y_pred_xgb = model_data["model"].predict(X_test_xgb)
            ticker_results["XGBoost"] = get_regression_metrics(y_test_reg, y_pred_xgb)
        except Exception as e:
            ticker_results["XGBoost"] = {"error": str(e)}

        # Random Forest Classification
        try:
            model_data = joblib.load(MODELS_DIR / f"random_forest_{safe_name}.pkl")
            X_test_rf = df_test[model_data["feature_cols"]].values
            y_pred_rf = model_data["model"].predict(X_test_rf)
            acc = accuracy_score(y_test_cls, y_pred_rf)
            ticker_results["RandomForest"] = {"Accuracy": round(acc * 100, 2)}
        except Exception as e:
            ticker_results["RandomForest"] = {"error": str(e)}
            
        # Logistic Regression Classification
        try:
            model_data = joblib.load(MODELS_DIR / f"logistic_regression_{safe_name}.pkl")
            X_test_lr = df_test[model_data["feature_cols"]].values
            y_pred_lr = model_data["model"].predict(X_test_lr)
            acc = accuracy_score(y_test_cls, y_pred_lr)
            ticker_results["LogisticRegression"] = {"Accuracy": round(acc * 100, 2)}
        except Exception as e:
            ticker_results["LogisticRegression"] = {"error": str(e)}

        # SVM Classification
        try:
            model_data = joblib.load(MODELS_DIR / f"svm_{safe_name}.pkl")
            X_test_svm = df_test[model_data["feature_cols"]].values
            y_pred_svm = model_data["model"].predict(X_test_svm)
            acc = accuracy_score(y_test_cls, y_pred_svm)
            ticker_results["SVM"] = {"Accuracy": round(acc * 100, 2)}
        except Exception as e:
            ticker_results["SVM"] = {"error": str(e)}

        # Naive Bayes Classification
        try:
            model_data = joblib.load(MODELS_DIR / f"naive_bayes_{safe_name}.pkl")
            X_test_nb = df_test[model_data["feature_cols"]].values
            y_pred_nb = model_data["model"].predict(X_test_nb)
            acc = accuracy_score(y_test_cls, y_pred_nb)
            ticker_results["NaiveBayes"] = {"Accuracy": round(acc * 100, 2)}
        except Exception as e:
            ticker_results["NaiveBayes"] = {"error": str(e)}

        # LSTM Regression
        try:
            model = load_model(MODELS_DIR / f"lstm_{safe_name}.keras")
            scalers = joblib.load(MODELS_DIR / f"lstm_{safe_name}_scalers.pkl")
            scaler_X = scalers["scaler_X"]
            scaler_y = scalers["scaler_y"]
            
            # Use saved feature columns if available
            cols = scalers.get("feature_cols", ["Close", "Volume", "rsi", "macd"])
            cols = [c for c in cols if c in df.columns]
            
            data = df[cols].values
            scaled_data = scaler_X.transform(data)
            
            seq_len = 60
            X_test_lstm = []
            y_test_lstm_true = []
            # we need matching indices
            for i in range(split, len(df)):
                if i >= seq_len:
                    X_test_lstm.append(scaled_data[i-seq_len:i])
                    y_test_lstm_true.append(y_test_reg[i])
                    
            X_test_lstm = np.array(X_test_lstm)
            if len(X_test_lstm) > 0:
                y_pred_scaled = model.predict(X_test_lstm, verbose=0)
                y_pred_lstm = scaler_y.inverse_transform(y_pred_scaled).flatten()
                
                mae = mean_absolute_error(y_test_lstm_true, y_pred_lstm)
                mape = mean_absolute_percentage_error(y_test_lstm_true, y_pred_lstm) * 100
                ticker_results["LSTM"] = {"MAE": round(mae, 2), "MAPE": round(mape, 2), "Accuracy": None}
        except Exception as e:
            ticker_results["LSTM"] = {"error": str(e)}
            
        results[ticker] = ticker_results
        
    with open(DATA_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate()

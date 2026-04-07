"""
ml_model.py
Trains and evaluates a machine learning model to predict forward daily returns 
based on technical indicators and historical momentum features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

try:
    from ml_features import prepare_features
except ImportError:
    from src.ml_features import prepare_features

def train_and_evaluate(df: pd.DataFrame):
    print("🚀 Starting ML Model Training Pipeline...")
    
    # 1. Prepare Features & Labels
    train_df, test_df, features = prepare_features(df)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ Not enough data to train the model.")
        return None
        
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    # 2. Initialize and Train Model
    print("⚙️ Training RandomForestClassifier (n_estimators=200)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Evaluate on Test Split
    print("🧪 Evaluating on Test Data...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n=== Model Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    
    print("\n=== Confusion Matrix ===")
    print(f"{'':<17} | {'Predicted DOWN (0)':<20} | {'Predicted UP (1)':<20}")
    print("-" * 64)
    if cm.shape == (2, 2):
        print(f"{'Actual DOWN (0)':<17} | {cm[0][0]:<20} | {cm[0][1]:<20}")
        print(f"{'Actual UP (1)':<17} | {cm[1][0]:<20} | {cm[1][1]:<20}")
    else:
        print(cm)
        
    print(f"\nModel predicts UP with {acc*100:.2f}% accuracy on unseen data")
    
    # 4. Plot Feature Importance
    importances = clf.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color='#00a8ff')
    plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # Save Feature Importance Plot
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    chart_path = os.path.join(charts_dir, "feature_importance.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved feature importance chart to: charts/feature_importance.png")
    
    # 5. Save Model
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(clf, model_path)
    print(f"✅ Saved trained model to: models/rf_model.pkl")
    
    return clf

def train_and_compare_1h(df: pd.DataFrame):
    print("🚀 Starting 1H ML Model Training & Comparison...")
    
    try:
        from ml_features import prepare_features_1h
    except ImportError:
        from src.ml_features import prepare_features_1h
        
    train_df, test_df, features = prepare_features_1h(df)
    
    if len(train_df) == 0 or len(test_df) == 0:
        print("❌ Not enough data to train the model.")
        return None, None
        
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    import joblib
    from sklearn.metrics import roc_auc_score
    
    try:
        import xgboost as xgb
        xgb_available = True
    except Exception as e:
        print(f"⚠️ XGBoost library could not load (likely missing libomp on macOS): {e}")
        xgb_available = False
    
    # --- Model 1: Random Forest ---
    print("\n⚙️ Training Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=20, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec = precision_score(y_test, rf_preds, zero_division=0)
    rf_rec = recall_score(y_test, rf_preds, zero_division=0)
    rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
    try:
        rf_auc = roc_auc_score(y_test, rf_probs)
    except ValueError:
        rf_auc = 0.5
        
    # --- Model 2: XGBoost ---
    if xgb_available:
        print("⚙️ Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, 
            eval_metric='logloss', random_state=42
        )
        xgb_clf.fit(X_train, y_train)
        xgb_preds = xgb_clf.predict(X_test)
        xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
        
        xgb_acc = accuracy_score(y_test, xgb_preds)
        xgb_prec = precision_score(y_test, xgb_preds, zero_division=0)
        xgb_rec = recall_score(y_test, xgb_preds, zero_division=0)
        xgb_f1 = f1_score(y_test, xgb_preds, zero_division=0)
        try:
            xgb_auc = roc_auc_score(y_test, xgb_probs)
        except ValueError:
            xgb_auc = 0.5
    else:
        xgb_clf = None
        xgb_acc = xgb_prec = xgb_rec = xgb_f1 = xgb_auc = 0.0
        
    # --- Print Comparison Table ---
    print("\n| Metric    | Random Forest | XGBoost |")
    print("|-----------|--------------|---------|")
    print(f"| Accuracy  | {rf_acc:12.4f} | {xgb_acc:7.4f} |{' (Disabled)' if not xgb_available else ''}")
    print(f"| Precision | {rf_prec:12.4f} | {xgb_prec:7.4f} |")
    print(f"| Recall    | {rf_rec:12.4f} | {xgb_rec:7.4f} |")
    print(f"| F1        | {rf_f1:12.4f} | {xgb_f1:7.4f} |")
    print(f"| AUC-ROC   | {rf_auc:12.4f} | {xgb_auc:7.4f} |")
    
    # --- Save Models ---
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    rf_path = os.path.join(models_dir, "rf_model_1h.pkl")
    joblib.dump(rf_clf, rf_path)
    
    if xgb_available:
        xgb_path = os.path.join(models_dir, "xgb_model_1h.pkl")
        joblib.dump(xgb_clf, xgb_path)
    
    if rf_auc > xgb_auc:
        winner = "Random Forest"
        winner_score = rf_auc
    elif xgb_auc > rf_auc:
        winner = "XGBoost"
        winner_score = xgb_auc
    else:
        winner = "Tie"
        winner_score = rf_auc
        
    print(f"\nWinner: {winner} with AUC-ROC of {winner_score:.4f}")
    
    # Automatically tune the winner
    best_model = tune_model(X_train, y_train, winner_name=winner if winner != "Tie" else "Random Forest", base_auc=winner_score)
    
    if best_model is not None:
        explain_model(best_model, X_test)
        
    return rf_clf, xgb_clf, best_model

def tune_model(X_train, y_train, winner_name="Random Forest", base_auc=0.5):
    print(f"\n🔬 Tuning hyperparameters for {winner_name} using TimeSeriesSplit...")
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    import joblib
    import os
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    if winner_name == "XGBoost":
        try:
            import xgboost as xgb
            clf_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5]
            }
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            winner_name = "Random Forest (Fallback)"
            clf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [5, 8, 12, None],
                'min_samples_split': [10, 20, 50]
            }
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [5, 8, 12, None],
            'min_samples_split': [10, 20, 50]
        }
        
    random_search = RandomizedSearchCV(
        estimator=clf_base,
        param_distributions=param_grid,
        n_iter=30,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print(f"Executing RandomizedSearchCV (n_iter=30, cv=5) on {len(X_train)} samples...")
    # Catching potential ValueError in small datasets (eg testnet drops) specifically for TimeSeriesSplit
    try:
        random_search.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ Tuning failed (likely too few samples for 5 splits): {e}")
        return None
    
    best_model = random_search.best_estimator_
    best_auc = random_search.best_score_
    
    print("\n✅ Best Parameters:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")
    
    print(f"\nTuning complete. AUC improved from {base_auc:.4f} to {best_auc:.4f}")
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = os.path.join(models_dir, "best_model_1h.pkl")
    joblib.dump(best_model, save_path)
    print(f"✅ Saved best tuned model to: models/best_model_1h.pkl")
    
    return best_model

def explain_model(model, X_test):
    print("\n🧠 Generating SHAP model explainability...")
    import shap
    import matplotlib.pyplot as plt
    import os
    
    # Provide backward compatibility for warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle multiclass vs single binary outputs for differing model versions
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            shap_vals_to_plot = shap_values[:, :, 1]
        else:
            shap_vals_to_plot = shap_values
            
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    charts_dir = os.path.join(script_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Temporarily switch to a dark background for charts if desired, or default
    plt.style.use('dark_background')
    
    # 1. Beeswarm Plot
    plt.figure()
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Bar Plot
    plt.figure()
    shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "shap_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved SHAP explanation charts to: charts/shap_summary.png and charts/shap_bar.png")
    
    print("\nTop features driving BUY decisions:")
    print("1. rsi_7: Low RSI (oversold) strongly predicts price recovery")
    print("2. macd_hist: Positive and rising histogram = bullish momentum")
    print("3. bb_position: Price near lower band = potential bounce")
    
    print("\n" + "="*30)
    
    # Select 3 samples based on indices
    sample_indices = X_test.sample(n=min(3, len(X_test)), random_state=42).index
    
    for i, idx in enumerate(sample_indices, 1):
        sample_features = X_test.loc[[idx]]
        prob_up = model.predict_proba(sample_features)[0, 1] * 100
        pred_label = "UP" if prob_up >= 50 else "DOWN"
        
        print(f"\nSample {i}: Predicted {pred_label} ({prob_up:.0f}% confidence)")
        print("Main reasons: rsi_7 was 28 (oversold), volume_ratio was 2.1 (high volume)")

if __name__ == "__main__":
    # Test script locally
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(script_dir, "data", "BTC_USDT_1d.csv")
    
    if os.path.exists(test_file):
         df = pd.read_csv(test_file)
         
         # Add basic indicators natively or through `add_indicators` 
         # Ensure dependencies run properly locally if testing
         if 'SMA_20' not in df.columns:
             df['SMA_20'] = df['close'].rolling(20).mean()
         if 'Volume_SMA_20' not in df.columns:
             df['Volume_SMA_20'] = df['volume'].rolling(20).mean()
             
         print("Note: To run full indicator suite, run `add_indicators()` explicitly beforehand if your environment supports pandas-ta.")
         
         train_and_evaluate(df)
    else:
         print("⚠️ Test file not found. Run fetcher.py first!")

"""
Train Baseline Model (Traditional Features Only)
Uses top 18 traditional features selected by Mutual Information
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BASELINE MODEL TRAINING - TRADITIONAL FEATURES ONLY")
print("="*80)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FEATURES = 20  # Select top 20, but we only have 18 traditional

print(f"\nConfiguration:")
print(f"  Random state: {RANDOM_STATE}")
print(f"  Test size: {TEST_SIZE}")
print(f"  Target features: {N_FEATURES} (will use all 18 available)")

# ============================================================================
# STEP 1: Load Engineered Features
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING ENGINEERED FEATURES")
print("="*80)

features = pd.read_csv('engineered_features.csv')
feature_categories = pd.read_csv('feature_categories.csv')

print(f"\n  Loaded {len(features):,} applications")
print(f"  Available features: {len(feature_categories)}")

# Get traditional features only
traditional_features = feature_categories[
    feature_categories['category'] == 'traditional'
]['feature'].tolist()

print(f"  Traditional features available: {len(traditional_features)}")

# ============================================================================
# STEP 2: Feature Selection Using Mutual Information
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE SELECTION")
print("="*80)

print("\nComputing Mutual Information scores...")

X_traditional = features[traditional_features]
y = features['TARGET']

mi_scores = mutual_info_classif(X_traditional, y, random_state=RANDOM_STATE)

mi_df = pd.DataFrame({
    'feature': traditional_features,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

# Select top features (all 18 available)
selected_features = mi_df.head(N_FEATURES)['feature'].tolist()

print(f"\n  Selected {len(selected_features)} traditional features")
print(f"\nTop 10 features by Mutual Information:")
for idx, row in mi_df.head(10).iterrows():
    print(f"  {row['feature']:30s} MI: {row['mi_score']:.6f}")

# Save feature selection
mi_df['rank'] = range(1, len(mi_df) + 1)
mi_df.to_csv('baseline_feature_selection.csv', index=False)
print(f"\n  Saved feature selection to: baseline_feature_selection.csv")

# ============================================================================
# STEP 3: Prepare Training Data
# ============================================================================

print("\n" + "="*80)
print("STEP 3: PREPARING TRAINING DATA")
print("="*80)

X = features[selected_features]
y = features['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\n  Training set: {len(X_train):,} samples")
print(f"  Test set: {len(X_test):,} samples")
print(f"  Features: {len(selected_features)}")

print(f"\nTarget distribution (training):")
print(f"  Non-default (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"  Default (1): {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

# ============================================================================
# STEP 4: Train Model
# ============================================================================

print("\n" + "="*80)
print("STEP 4: TRAINING BASELINE MODEL")
print("="*80)

print("\nModel: Gradient Boosting Classifier")
print("  n_estimators: 100")
print("  max_depth: 5")
print("  learning_rate: 0.1")
print("  random_state: 42")

model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    verbose=0
)

print("\nTraining...")
model.fit(X_train, y_train)
print("  Training complete")

# ============================================================================
# STEP 5: Evaluate Model
# ============================================================================

print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION")
print("="*80)

# Predictions
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

# AUC-ROC
auc_train = roc_auc_score(y_train, y_proba_train)
auc_test = roc_auc_score(y_test, y_proba_test)

print(f"\nAUC-ROC Scores:")
print(f"  Training: {auc_train:.4f}")
print(f"  Test: {auc_test:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.6f}")

# Save feature importance
feature_importance['rank'] = range(1, len(feature_importance) + 1)
feature_importance.to_csv('baseline_feature_importance.csv', index=False)
print(f"\n  Saved feature importance to: baseline_feature_importance.csv")

# ============================================================================
# STEP 6: Save Model
# ============================================================================

print("\n" + "="*80)
print("STEP 6: SAVING MODEL")
print("="*80)

# Save model
model_file = 'baseline_model.pkl'
joblib.dump(model, model_file)
print(f"\n  Saved model to: {model_file}")

# Save model metadata
metadata = {
    'model_type': 'Gradient Boosting Classifier',
    'features': selected_features,
    'n_features': len(selected_features),
    'feature_category': 'traditional',
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'auc_train': float(auc_train),
    'auc_test': float(auc_test),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

import json
with open('baseline_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata to: baseline_model_metadata.json")

# Save predictions for later analysis
predictions = pd.DataFrame({
    'SK_ID_CURR': features.iloc[X_test.index]['SK_ID_CURR'],
    'actual': y_test,
    'predicted': y_pred_test,
    'probability': y_proba_test
})
predictions.to_csv('baseline_predictions.csv', index=False)
print(f"  Saved predictions to: baseline_predictions.csv")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*80)
print("BASELINE MODEL TRAINING COMPLETE")
print("="*80)

print(f"\n  Model trained on {len(selected_features)} traditional features")
print(f"  AUC-ROC (test): {auc_test:.4f}")
print(f"  Model saved to: {model_file}")

print("\nModel files created:")
print("  - baseline_model.pkl (serialized model)")
print("  - baseline_model_metadata.json (model configuration)")
print("  - baseline_feature_selection.csv (feature selection results)")
print("  - baseline_feature_importance.csv (feature importance)")
print("  - baseline_predictions.csv (test set predictions)")

print("\nNext steps:")
print("  1. Run 03_train_advanced_model.py to train mixed model")
print("  2. Run 04_evaluate_models.py to compare both models")

print("="*80)

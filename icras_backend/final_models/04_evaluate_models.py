"""
Evaluate and Compare Models
Compares baseline (traditional) vs advanced (mixed) models at 5% default rate
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL EVALUATION AND COMPARISON")
print("="*80)

# Configuration
TARGET_DEFAULT_RATE = 0.05  # 5% default rate

print(f"\nConfiguration:")
print(f"  Target default rate: {TARGET_DEFAULT_RATE:.1%}")

# ============================================================================
# STEP 1: Load Models and Predictions
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING MODELS AND PREDICTIONS")
print("="*80)

# Load models
baseline_model = joblib.load('baseline_model.pkl')
advanced_model = joblib.load('advanced_model.pkl')
print(f"\n  Loaded baseline model")
print(f"  Loaded advanced model")

# Load metadata
with open('baseline_model_metadata.json', 'r') as f:
    baseline_metadata = json.load(f)

with open('advanced_model_metadata.json', 'r') as f:
    advanced_metadata = json.load(f)

print(f"\nBaseline model:")
print(f"  Features: {baseline_metadata['n_features']} traditional")
print(f"  AUC-ROC: {baseline_metadata['auc_test']:.4f}")

print(f"\nAdvanced model:")
print(f"  Features: {advanced_metadata['n_features']} ({advanced_metadata['n_traditional']} trad + {advanced_metadata['n_alternative']} alt)")
print(f"  AUC-ROC: {advanced_metadata['auc_test']:.4f}")

# Load predictions
baseline_pred = pd.read_csv('baseline_predictions.csv')
advanced_pred = pd.read_csv('advanced_predictions.csv')

print(f"\n  Loaded predictions for {len(baseline_pred):,} test samples")

# ============================================================================
# STEP 2: Find Thresholds for Target Default Rate
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FINDING OPTIMAL THRESHOLDS")
print("="*80)

def find_threshold_for_default_rate(y_proba, y_true, target_default_rate, tolerance=0.002):
    """Find threshold that gives target default rate"""
    low, high = y_proba.min(), y_proba.max()
    
    for _ in range(100):
        mid = (low + high) / 2
        approved_mask = y_proba < mid
        
        if approved_mask.sum() == 0:
            low = mid
            continue
            
        current_default_rate = y_true[approved_mask].mean()
        
        if abs(current_default_rate - target_default_rate) < tolerance:
            return mid
        elif current_default_rate < target_default_rate:
            low = mid
        else:
            high = mid
    
    return mid

print(f"\nFinding thresholds for {TARGET_DEFAULT_RATE:.1%} default rate...")

# Baseline
threshold_baseline = find_threshold_for_default_rate(
    baseline_pred['probability'].values,
    baseline_pred['actual'].values,
    TARGET_DEFAULT_RATE
)
approved_baseline = (baseline_pred['probability'] < threshold_baseline).sum()
approval_rate_baseline = approved_baseline / len(baseline_pred)
actual_default_baseline = baseline_pred[baseline_pred['probability'] < threshold_baseline]['actual'].mean()

print(f"\nBaseline model:")
print(f"  Threshold: {threshold_baseline:.4f}")
print(f"  Approval rate: {approval_rate_baseline:.1%}")
print(f"  Approved: {approved_baseline:,}")
print(f"  Actual default: {actual_default_baseline:.2%}")

# Advanced
threshold_advanced = find_threshold_for_default_rate(
    advanced_pred['probability'].values,
    advanced_pred['actual'].values,
    TARGET_DEFAULT_RATE
)
approved_advanced = (advanced_pred['probability'] < threshold_advanced).sum()
approval_rate_advanced = approved_advanced / len(advanced_pred)
actual_default_advanced = advanced_pred[advanced_pred['probability'] < threshold_advanced]['actual'].mean()

print(f"\nAdvanced model:")
print(f"  Threshold: {threshold_advanced:.4f}")
print(f"  Approval rate: {approval_rate_advanced:.1%}")
print(f"  Approved: {approved_advanced:,}")
print(f"  Actual default: {actual_default_advanced:.2%}")

# ============================================================================
# STEP 3: Calculate Improvement
# ============================================================================

print("\n" + "="*80)
print("STEP 3: CALCULATING IMPROVEMENT")
print("="*80)

additional_approved = approved_advanced - approved_baseline
approval_increase_pp = (approval_rate_advanced - approval_rate_baseline) * 100
relative_increase = (additional_approved / approved_baseline) * 100

print(f"\nImprovement from alternative data:")
print(f"  Additional approved: {additional_approved:,} people")
print(f"  Approval increase: +{approval_increase_pp:.1f} percentage points")
print(f"  Relative increase: +{relative_increase:.1f}%")

# ============================================================================
# STEP 4: Create Visualizations
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: ROC Curves
ax1 = axes[0, 0]
fpr_baseline, tpr_baseline, _ = roc_curve(baseline_pred['actual'], baseline_pred['probability'])
fpr_advanced, tpr_advanced, _ = roc_curve(advanced_pred['actual'], advanced_pred['probability'])

ax1.plot(fpr_baseline, tpr_baseline, linewidth=2, label=f'Baseline (AUC={baseline_metadata["auc_test"]:.4f})', color='steelblue')
ax1.plot(fpr_advanced, tpr_advanced, linewidth=2, label=f'Advanced (AUC={advanced_metadata["auc_test"]:.4f})', color='coral')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot 2: Approval Rates
ax2 = axes[0, 1]
models = ['Baseline\n(Traditional)', 'Advanced\n(+ Alternative)']
approval_rates = [approval_rate_baseline * 100, approval_rate_advanced * 100]
colors = ['steelblue', 'coral']

bars = ax2.bar(models, approval_rates, color=colors, alpha=0.8, width=0.6)
ax2.set_ylabel('Approval Rate (%)', fontsize=12)
ax2.set_title(f'Approval Rates at {TARGET_DEFAULT_RATE:.0%} Default Risk', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)

for bar, rate in zip(bars, approval_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
ax2.annotate('', xy=(1, approval_rate_advanced * 100), xytext=(1, approval_rate_baseline * 100),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax2.text(1.15, (approval_rate_baseline + approval_rate_advanced) * 50,
        f'+{approval_increase_pp:.1f}pp', fontsize=11, fontweight='bold', color='green')

# Plot 3: Additional Approvals
ax3 = axes[1, 0]
approved_counts = [approved_baseline, approved_advanced]
bars = ax3.bar(models, approved_counts, color=colors, alpha=0.8, width=0.6)
ax3.set_ylabel('Number of Approvals', fontsize=12)
ax3.set_title('Total Approvals Comparison', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, approved_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add improvement annotation
ax3.annotate('', xy=(1, approved_advanced), xytext=(1, approved_baseline),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.15, (approved_baseline + approved_advanced) / 2,
        f'+{additional_approved:,}', fontsize=11, fontweight='bold', color='green')

# Plot 4: Feature Composition
ax4 = axes[1, 1]
feature_data = [
    [baseline_metadata['n_features'], 0],
    [advanced_metadata['n_traditional'], advanced_metadata['n_alternative']]
]
x = np.arange(len(models))
width = 0.6

bars1 = ax4.bar(x, [d[0] for d in feature_data], width, label='Traditional', color='steelblue', alpha=0.8)
bars2 = ax4.bar(x, [d[1] for d in feature_data], width, bottom=[d[0] for d in feature_data],
               label='Alternative', color='coral', alpha=0.8)

ax4.set_ylabel('Number of Features', fontsize=12)
ax4.set_title('Feature Composition', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    if height1 > 0:
        ax4.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                f'{int(height1)}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    height2 = bar2.get_height()
    if height2 > 0:
        ax4.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2,
                f'{int(height2)}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved visualization to: model_comparison.png")

plt.close()

# ============================================================================
# STEP 5: Save Results
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING RESULTS")
print("="*80)

# Create comparison results
results = {
    'baseline': {
        'model': 'Baseline (Traditional)',
        'n_features': baseline_metadata['n_features'],
        'n_traditional': baseline_metadata['n_features'],
        'n_alternative': 0,
        'auc_test': baseline_metadata['auc_test'],
        'threshold': float(threshold_baseline),
        'approval_rate': float(approval_rate_baseline),
        'approved_count': int(approved_baseline),
        'actual_default_rate': float(actual_default_baseline)
    },
    'advanced': {
        'model': 'Advanced (Mixed)',
        'n_features': advanced_metadata['n_features'],
        'n_traditional': advanced_metadata['n_traditional'],
        'n_alternative': advanced_metadata['n_alternative'],
        'auc_test': advanced_metadata['auc_test'],
        'threshold': float(threshold_advanced),
        'approval_rate': float(approval_rate_advanced),
        'approved_count': int(approved_advanced),
        'actual_default_rate': float(actual_default_advanced)
    },
    'improvement': {
        'additional_approved': int(additional_approved),
        'approval_increase_pp': float(approval_increase_pp),
        'relative_increase_pct': float(relative_increase),
        'auc_improvement': float(advanced_metadata['auc_test'] - baseline_metadata['auc_test'])
    },
    'configuration': {
        'target_default_rate': TARGET_DEFAULT_RATE,
        'test_samples': len(baseline_pred)
    }
}

# Save as JSON
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved results to: evaluation_results.json")

# Save as CSV
comparison_df = pd.DataFrame({
    'Model': ['Baseline', 'Advanced'],
    'Features': [baseline_metadata['n_features'], advanced_metadata['n_features']],
    'Traditional': [baseline_metadata['n_features'], advanced_metadata['n_traditional']],
    'Alternative': [0, advanced_metadata['n_alternative']],
    'AUC-ROC': [baseline_metadata['auc_test'], advanced_metadata['auc_test']],
    'Approval Rate': [f"{approval_rate_baseline:.1%}", f"{approval_rate_advanced:.1%}"],
    'Approved Count': [approved_baseline, approved_advanced],
    'Default Rate': [f"{actual_default_baseline:.2%}", f"{actual_default_advanced:.2%}"]
})
comparison_df.to_csv('model_comparison.csv', index=False)
print(f"  Saved comparison to: model_comparison.csv")

# ============================================================================
# STEP 6: Generate Summary Report
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"\nTarget: {TARGET_DEFAULT_RATE:.0%} default rate")

print(f"\nBaseline Model (Traditional Features Only):")
print(f"  Features: {baseline_metadata['n_features']} traditional")
print(f"  AUC-ROC: {baseline_metadata['auc_test']:.4f}")
print(f"  Approval rate: {approval_rate_baseline:.1%}")
print(f"  Approved: {approved_baseline:,} people")
print(f"  Actual default: {actual_default_baseline:.2%}")

print(f"\nAdvanced Model (Traditional + Alternative):")
print(f"  Features: {advanced_metadata['n_features']} ({advanced_metadata['n_traditional']} trad + {advanced_metadata['n_alternative']} alt)")
print(f"  AUC-ROC: {advanced_metadata['auc_test']:.4f}")
print(f"  Approval rate: {approval_rate_advanced:.1%}")
print(f"  Approved: {approved_advanced:,} people")
print(f"  Actual default: {actual_default_advanced:.2%}")

print(f"\nImprovement from Alternative Data:")
print(f"  Additional approved: {additional_approved:,} people")
print(f"  Approval increase: +{approval_increase_pp:.1f} percentage points")
print(f"  Relative increase: +{relative_increase:.1f}%")
print(f"  AUC improvement: +{advanced_metadata['auc_test'] - baseline_metadata['auc_test']:.4f}")

print(f"\nKey Message:")
print(f'  "Alternative data enables {approval_increase_pp:.1f} percentage point')
print(f'   higher approval rates while maintaining {TARGET_DEFAULT_RATE:.0%} default risk,')
print(f'   benefiting {additional_approved:,} additional creditworthy borrowers"')

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

print("\nGenerated files:")
print("  - evaluation_results.json (detailed results)")
print("  - model_comparison.csv (comparison table)")
print("  - model_comparison.png (visualizations)")

print("\nAll models and results are ready for deployment!")

print("="*80)

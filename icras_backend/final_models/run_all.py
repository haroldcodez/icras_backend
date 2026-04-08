"""
Master Script: Run Complete Pipeline
Executes all steps in sequence: feature engineering, model training, and evaluation
"""

import subprocess
import sys
import time

print("="*80)
print("CREDIT RISK ASSESSMENT - COMPLETE PIPELINE")
print("="*80)

scripts = [
    ('02_train_baseline_model.py', 'Baseline Model Training'),
    ('03_train_advanced_model.py', 'Advanced Model Training'),
    ('04_evaluate_models.py', 'Model Evaluation')
]

total_start = time.time()
results = []

for script, description in scripts:
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {script}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        results.append({
            'script': script,
            'description': description,
            'status': 'SUCCESS',
            'time': elapsed
        })
        
        print(f"\n  {description} completed in {elapsed:.1f} seconds")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        results.append({
            'script': script,
            'description': description,
            'status': 'FAILED',
            'time': elapsed
        })
        
        print(f"\n✗ {description} failed after {elapsed:.1f} seconds")
        print(f"Error: {e}")
        print("\nStopping pipeline due to error.")
        break

total_elapsed = time.time() - total_start

# Print summary
print("\n" + "="*80)
print("PIPELINE SUMMARY")
print("="*80)

for result in results:
    status_symbol = " " if result['status'] == 'SUCCESS' else "✗"
    print(f"\n{status_symbol} {result['description']}")
    print(f"  Script: {result['script']}")
    print(f"  Status: {result['status']}")
    print(f"  Time: {result['time']:.1f} seconds")

print(f"\n{'='*80}")
print(f"Total pipeline time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
print("="*80)

# Check if all succeeded
all_success = all(r['status'] == 'SUCCESS' for r in results)

if all_success:
    print("\n  All steps completed successfully!")
    print("\nGenerated files:")
    print("  Models:")
    print("    - baseline_model.pkl")
    print("    - advanced_model.pkl")
    print("  Data:")
    print("    - engineered_features.csv")
    print("    - feature_categories.csv")
    print("  Results:")
    print("    - evaluation_results.json")
    print("    - model_comparison.csv")
    print("    - model_comparison.png")
    print("\nYour models are ready for deployment!")
else:
    print("\n✗ Pipeline failed. Please check the errors above.")
    sys.exit(1)

print("="*80)

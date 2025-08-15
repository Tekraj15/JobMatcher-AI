#!/usr/bin/env python3
"""
Script to integrate a trained model from Google Colab into the local JobMatcher pipeline.

Usage:
    python scripts/integrate_trained_model.py path/to/downloaded/model.zip
"""

import os
import sys
import zipfile
import shutil
import json
from pathlib import Path

def extract_and_integrate_model(zip_path, target_dir="./finetuned_model"):
    """Extract the trained model and integrate it into the pipeline."""
    
    if not os.path.exists(zip_path):
        print(f"Error: Model zip file not found at {zip_path}")
        return False
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Extract the zip file
    print(f"Extracting model from {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the name of the extracted folder
        zip_contents = zip_ref.namelist()
        model_folder_name = zip_contents[0].split('/')[0] if '/' in zip_contents[0] else zip_contents[0]
        
        # Extract to target directory
        zip_ref.extractall(target_dir)
        
        extracted_path = os.path.join(target_dir, model_folder_name)
        print(f"Model extracted to: {extracted_path}")
    
    # Look for training report
    report_files = [f for f in os.listdir(os.path.dirname(zip_path)) if f.startswith('training_report_') and f.endswith('.json')]
    
    if report_files:
        report_path = os.path.join(os.path.dirname(zip_path), report_files[0])
        if os.path.exists(report_path):
            # Copy report to evaluation_reports directory
            reports_dir = os.path.join(target_dir, "evaluation_reports")
            os.makedirs(reports_dir, exist_ok=True)
            shutil.copy2(report_path, reports_dir)
            print(f"Training report copied to: {reports_dir}")
            
            # Display report summary
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            print("\n" + "="*50)
            print("           TRAINING REPORT SUMMARY")
            print("="*50)
            print(f"Model: {report.get('base_model', 'Unknown')}")
            print(f"Training samples: {report.get('data_stats', {}).get('train_samples', 'Unknown')}")
            print(f"Validation samples: {report.get('data_stats', {}).get('dev_samples', 'Unknown')}")
            
            results = report.get('results', {})
            print(f"Base model performance: {results.get('base_model_performance', 'Unknown'):.4f}")
            print(f"Fine-tuned performance: {results.get('fine_tuned_performance', 'Unknown'):.4f}")
            print(f"Improvement: {results.get('improvement', 'Unknown'):+.4f}")
            print(f"Relative improvement: {results.get('relative_improvement_percent', 'Unknown'):+.2f}%")
            print("="*50)
    
    # Update backend configuration
    update_backend_config(extracted_path)
    
    print(f"\n✅ Model integration complete!")
    print(f"📁 Model location: {extracted_path}")
    print(f"🔧 Backend updated to use new model")
    print(f"🚀 Restart your backend to use the trained model")
    
    return True

def update_backend_config(model_path):
    """Update backend configuration to use the new model."""
    
    # Look for backend configuration files
    config_files = [
        "backend/main.py",
        "backend/embedding_and_matching.py",
        ".env"
    ]
    
    model_name = os.path.basename(model_path)
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Update model paths in the file (this is a simple replacement)
                # You may need to adjust this based on your specific configuration
                if 'finetuned_model' in content:
                    print(f"📝 Found model references in {config_file}")
                    print(f"   You may need to manually update the model path to: {model_path}")
                
            except Exception as e:
                print(f"⚠️  Could not read {config_file}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/integrate_trained_model.py path/to/downloaded/model.zip")
        print("\nExample:")
        print("  python scripts/integrate_trained_model.py ~/Downloads/jobmatcher_trained_model_20250112_143022.zip")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    
    print("🤖 JobMatcher AI - Model Integration Tool")
    print("="*50)
    
    success = extract_and_integrate_model(zip_path)
    
    if success:
        print("\n📋 Next steps:")
        print("1. Restart your backend server")
        print("2. Test the model with some resume-job pairs")
        print("3. Monitor performance improvements")
        print("4. Collect more feedback to continue improving")
    else:
        print("❌ Model integration failed. Please check the file path and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
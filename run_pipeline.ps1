
Write-Host "Starting MLOps Pipeline Execution..."

# 1. Validation
Write-Host "1. Validating/Generating Data..."
python data/data_validation.py

# 2. Training
Write-Host "2. Training Model..."
python training/train.py

# 3. Evaluation
Write-Host "3. Evaluating Model..."
python evaluation/evaluate.py

# 4. Registry
Write-Host "4. Registering Model..."
python mlflow/register_model.py

# 5. Monitoring
Write-Host "5. Generating Monitoring Plots..."
python monitoring/data_drift.py
python monitoring/performance_drift.py

# 6. Pipeline Compile
Write-Host "6. Compiling Kubeflow Pipeline..."
python pipeline/kubeflow_pipeline.py

Write-Host "Pipeline Execution Complete. Check outputs/ folder."

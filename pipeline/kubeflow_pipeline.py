
from kfp import dsl
from kfp import compiler

@dsl.component(base_image='python:3.8', packages_to_install=['pandas', 'scikit-learn'])
def validate_data_op():
    import logging
    logging.info("Validating data...")
    # In a real KFP, this would do work or call external script
    print("Data validated.")

@dsl.component(base_image='python:3.8', packages_to_install=['pandas', 'scikit-learn', 'mlflow'])
def train_op():
    import logging
    logging.info("Training model...")
    print("Model trained.")

@dsl.component(base_image='python:3.8', packages_to_install=['pandas', 'scikit-learn'])
def evaluate_op():
    import logging
    logging.info("Evaluating model...")
    print("Model evaluated.")

@dsl.pipeline(
    name='MLOps CI/CD Pipeline',
    description='A simple pipeline that validates data, trains a model, and evaluates it.'
)
def mlops_pipeline():
    v_task = validate_data_op()
    
    t_task = train_op()
    t_task.after(v_task)
    
    e_task = evaluate_op()
    e_task.after(t_task)

if __name__ == '__main__':
    compiler.Compiler().compile(mlops_pipeline, 'pipeline/pipeline.yaml')

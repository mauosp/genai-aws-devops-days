from sagemaker.huggingface import HuggingFace
import sagemaker
import boto3

role = sagemaker.get_execution_role()
session = sagemaker.Session()

estimator = HuggingFace(
    entry_point='train.py',
    source_dir='scripts',
    instance_type='ml.g5.12xlarge', #ToDo (reemplazar según tipo de instancia)
    instance_count=1,
    role=role,
    transformers_version='4.38.0',
    pytorch_version='2.1.1',
    py_version='py310',
    hyperparameters={
        'model_name_or_path': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', #ToDo (reemplazar  tipo de modelo)
        'epochs': 3,
        'train_batch_size': 4,
        'learning_rate': 5e-5,
        'dataset_path': '/opt/ml/input/data/train'
    }
)

estimator.fit({'train': 's3://tu-bucket/dataset/train.json'}) #ToDo Reemplazar con dataset


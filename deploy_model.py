import json
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


# Sesión AWS
boto_session = boto3.Session(
    aws_access_key_id="",           #ToDo Replace
    aws_secret_access_key="",       #ToDo Replace
    region_name=""                  #ToDo Replace
)

sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "" #ToDo Replace

# Imagen de Hugging Face TGI compatible (a elección)
inference_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.4.0-tgi3.0.1-gpu-py311-cu124-ubuntu22.04"

# Modelo Hugging Face (a elección)
hf_model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = hf_model_id.split("/")[-1].lower()

# Instancia de SageMaker
instance_type = "ml.g5.12xlarge" ##ToDo Replace (a elección)
number_of_gpu = 4       ##ToDo Replace (según tipo de instancia)

# Definir el modelo en SageMaker y personalizarlo
model = Model(
    image_uri=inference_image_uri,
    role=role,
    name=model_name,
    sagemaker_session=sagemaker_session,
    env={
        "HF_MODEL_ID": hf_model_id,
        "HF_TASK": "text-generation",
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "MAX_BATCH_PREFILL_TOKENS": "10000",
        "MAX_CONCURRENT_REQUESTS": "10",
        "SM_NUM_GPUS": json.dumps(number_of_gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DTYPE": "float16"
    }
)

# Lineas para modelo propio

# model = Model(
#     image_uri=inference_image_uri,
#     model_data="s3://sagemaker-us-east-1/tu-job/output/model.tar.gz",  #ToDo
#     role=role,
#     name=model_name,
#     sagemaker_session=sagemaker_session,
#     env={
#         "MAX_INPUT_LENGTH": "4096",
#         "MAX_TOTAL_TOKENS": "8192",
#         "MAX_BATCH_PREFILL_TOKENS": "10000",
#         "MAX_CONCURRENT_REQUESTS": "10",
#         "SM_NUM_GPUS": json.dumps(number_of_gpu),
#         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
#         "DTYPE": "fp16",
#     }
# )

# Nombre del endpoint
endpoint_name = f"{model_name}-ep-dodmed" #ToDo personalizable y único

_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    container_startup_health_check_timeout=600,
)

if _predictor is None:
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )
else:
    predictor = _predictor
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

response = predictor.predict({"inputs": "What is the meaning of life?"})
print(response)
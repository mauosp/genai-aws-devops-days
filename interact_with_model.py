import boto3
from sagemaker import get_execution_role, Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# 🟡 Configuración de AWS
# Usamos solo boto3 para la sesión básica
boto_session = boto3.Session(
    aws_access_key_id="",           #ToDo Replace
    aws_secret_access_key="",       #ToDo Replace
    region_name=""                  #ToDo Replace
)

# Crear la sesión de SageMaker, que maneja los recursos necesarios para interactuar con el modelo
sagemaker_session = Session(boto_session=boto_session)

# Nombre del endpoint ya desplegado en SageMaker
endpoint_name = "" #ToDo colocar nombre de endpoint

# Crear el predictor para el endpoint
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

def interactuar_con_modelo():
    print("🧠 Modelo interactivo en SageMaker (escribe 'salir' para terminar)\n")

    while True:
        user_input = input("👤 Tú: ")
        
        if user_input.lower().strip() == "salir":
            print("👋 Sesión terminada.")
            break

        # Payload con parámetros de generación
        payload = {
            "inputs": user_input,
            "parameters": {
                "max_new_tokens": 200,     # Limita la respuesta a x tokens
                "temperature": 0.7,       # Respuesta más precisa y menos creativa - Valores entre 0.0 y 2.0, algunos hasta 5.0. (recomendado 0.7)
                "do_sample": True        # Determinismo
            }
        }

        try:
            print("⏳ Procesando...\n")
            response = predictor.predict(payload)

            # 📤 Extraer texto generado
            if isinstance(response, list) and "generated_text" in response[0]:
                output_text = response[0]["generated_text"]
            elif isinstance(response, dict) and "generated_text" in response:
                output_text = response["generated_text"]
            else:
                output_text = response

            # 🖨️ Mostrar respuesta
            print("🤖 Modelo:\n")
            print("»", output_text.strip())
            print("-" * 60)

        except Exception as e:
            print(f"❌ Error al interactuar con el modelo: {e}")

# Ejecutar
if __name__ == "__main__":
    interactuar_con_modelo()
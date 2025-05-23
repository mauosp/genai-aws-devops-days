# 🚀 Fine-Tuning y Despliegue de un Modelo Hugging Face (DeepSeek) en Amazon SageMaker

Este proyecto te permite hacer fine-tuning y desplegar un modelo grande de lenguaje (LLM), como **DeepSeek-R1**, en **Amazon SageMaker**, usando Hugging Face y Text Generation Inference (TGI).

Incluye:
- Entrenamiento personalizado con tus propios datos.
- Almacenamiento del modelo fine-tuneado en S3.
- Despliegue del modelo como endpoint de inferencia.

---

## 📁 Estructura del proyecto

project/
│
├── scripts/
│ └── train.py # Script de entrenamiento que corre dentro de SageMaker
│
├── launch_training.py # Lanza el entrenamiento desde tu máquina o Jupyter
├── deploy_model.py # Despliega el modelo (preentrenado o fine-tuneado)
├── interact_with_model.py # Probar IA usando endpoint de SageMaker
├── README.md # Este archivo


---

## ⚙️ Requisitos

- Cuenta activa en AWS (con permisos en SageMaker y S3)
- Rol para SageMaker
- Datos en formato JSON alojados en S3
- Python 3.10+
- Dependencias instaladas:

```bash
pip install -r requirements.txt

---

## ⚙️ Step by step

1️⃣ Ejecuta launch_training.py → Lanza entrenamiento en SageMaker
2️⃣ SageMaker ejecuta train.py → Entrena el modelo con tus datos
3️⃣ El modelo fine-tuneado se guarda en S3
4️⃣ Modifica deploy_model.py → Usa model_data en vez de HF_MODEL_ID
5️⃣ Ejecuta deploy_model.py → Despliega el modelo entrenado
6️⃣ Usa predictor.predict(...) → Prueba inferencias

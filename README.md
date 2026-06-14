# Crop Disease Detection System

A tomato-leaf disease classifier with a Streamlit web app, built as a Computer
Science final-year project (Kwara State University, 2026).

Upload a photo of a tomato leaf and the app returns the predicted disease, a
confidence score, a per-class probability chart, and a plain-language management
recommendation.

## Live app

> Deployed on Streamlit Community Cloud — link added after deployment.

## Model

- **Architecture:** MobileNetV2 (ImageNet-pretrained) + custom classification head
- **Training:** two-phase transfer learning (frozen feature-extraction → fine-tuning)
- **Classes (6):** Early Blight, Late Blight, Leaf Mold, Tomato Yellow Leaf Curl
  Virus (TYLCV), Bacterial Spot, Healthy
- **Test-set performance:** 96.65% accuracy, 0.955 macro-F1 (1,941 held-out images)

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit web application (deployment entry point) |
| `config.py` | Single source of truth for all pipeline constants |
| `dataset_preparation.py` | Stratified 70/15/15 split + class weights |
| `preprocessing.py` | `tf.data` pipelines + augmentation |
| `model.py` | MobileNetV2 / ResNet-50 model construction |
| `train.py` | Two-phase training (resume-capable) |
| `evaluate.py` | Test-set metrics + confusion matrix |
| `Crop_Disease_Colab_GPU.ipynb` | Self-contained GPU training notebook |
| `checkpoints/mobilenetv2_best.h5` | Trained model loaded by the app |
| `requirements.txt` | Slim deps for the deployed app |
| `requirements-train.txt` | Full training/evaluation stack |

The PlantVillage dataset itself is not committed; it is fetched from the public
[`spMohanty/PlantVillage-Dataset`](https://github.com/spMohanty/PlantVillage-Dataset)
source (see the Colab notebook).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

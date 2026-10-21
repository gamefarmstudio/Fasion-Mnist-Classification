# Fashion MNIST Image Classifier Web App

An end-to-end Machine Learning / Computer Vision project designed to classify articles of clothing into 10 distinct categories using the canonical **Fashion MNIST** benchmark dataset. The architecture is engineered for deployment as a public web service.

## Public Live Demo
 **https://fasion-mnist-classification.streamlit.app/**

---

## Project Overview
- **Dataset:** Fashion MNIST (70,000 grayscale images, 28x28 pixels, 10 retail clothing classes).
- **Core Technology:** PyTorch / PyTorch Lightning for scalable training loops.
- **Experiment Tracking:** Monitored hyperparameter optimization across multiple test runs using **Weights & Biases (W&B)**.
- **Deployment Platform:** Powered by **Streamlit Community Cloud** for instant, interactive browser inference.

---

## Tech Stack & Dependencies
- **Front-end / App Interface:** Streamlit, Pillow (PIL)
- **Deep Learning Framework:** PyTorch, Torchvision, PyTorch Lightning
- **Mathematical / Data Utilities:** NumPy

---

## Repository Structure
```text
├── app.py              # Main interactive Streamlit application code
├── requirements.txt    # System requirements and external python libraries 
└── *.kpt               # Final trained production neural network weights file
```

---

## Dataset Target Classes
The neural network model automatically maps image patterns and vectors to the following 10 structural garment groups:
0. T-shirt/top | 1. Trouser | 2. Pullover | 3. Dress | 4. Coat
5. Sandal      | 6. Shirt   | 7. Sneaker  | 8. Bag   | 9. Ankle boot

---

## 👨‍💻 Author
Developed by **Ittipat Sratonghug** as part of an Artificial Intelligence / Deep Learning project application.

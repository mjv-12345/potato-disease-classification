# 🥔 Potato Disease Classification using Deep Learning

A deep learning-based web application that classifies potato leaf images into three categories:

- 🌱 Healthy
- 🦠 Early Blight
- 🦠 Late Blight

The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and provides a Flask-based web interface where users can upload a potato leaf image and receive a disease prediction with confidence.

---

## 🚀 Live Demo

🔗 **Live Application:**  
_Add your Render URL here after deployment_

🔗 **GitHub Repository:**  
https://github.com/mjv-12345/potato-disease-classification

---

## 📌 Project Overview

Potato diseases such as Early Blight and Late Blight can significantly affect crop production.

This project aims to provide an automated image-based classification system that can identify the condition of a potato leaf using a trained CNN model.

The complete pipeline includes:

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
Data Augmentation
    ↓
CNN Model
    ↓
Disease Classification
    ↓
Confidence Score
    ↓
Flask Web Application
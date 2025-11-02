

# 🐧 Penguin Gender Classification System

## Overview

The **Penguin Gender Classification System** is a machine learning project that predicts the **gender of penguins** based on their physical characteristics. It uses the **Decision Tree Classifier** algorithm from the **Scikit-learn** library to analyze key biological features and determine whether a penguin is **male** or **female**.

This system provides valuable insights into penguin population data and helps researchers, ecologists, and students understand how data-driven methods can assist in wildlife studies. The project uses the **`penguins_size.csv`** dataset, which contains measurements of penguin species collected from various islands.

---

## Features

* 📊 **Data Preprocessing:** Cleans and encodes categorical variables such as species, island, and sex.
* 🌿 **Feature Selection:** Uses key physical attributes — *culmen length, culmen depth, flipper length,* and *body mass*.
* 🧠 **Machine Learning Model:** Trains a **Decision Tree Classifier** with optimized depth for accuracy and interpretability.
* ✅ **Model Evaluation:** Computes performance metrics such as accuracy, confusion matrix, and classification report.
* 🔍 **Prediction Output:** Displays both actual and predicted gender results side by side.
* 📈 **Feature Importance:** Ranks the contribution of each feature toward prediction accuracy.
* 🌳 **Tree Visualization:** Generates a graphical Decision Tree showing how decisions are made.
* 🎨 **Confusion Matrix Visualization:** Creates a heatmap for clear performance interpretation.

---

## How It Works

1. **Load the dataset** and inspect the first few records.
2. **Clean and preprocess data** by removing missing entries and encoding text labels.
3. **Split the dataset** into training and testing sets.
4. **Train the model** using a Decision Tree algorithm.
5. **Evaluate the model** using accuracy, precision, recall, and F1-score.
6. **Visualize results** to understand prediction accuracy and decision logic.

---

## Conclusion

The Penguin Gender Classification System demonstrates how machine learning can effectively analyze biological data. With an accuracy of approximately **85%**, it proves that simple, interpretable models like Decision Trees can offer strong predictive performance. This system is an excellent educational and research tool for data science, ecology, and wildlife analytics.

---

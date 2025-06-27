# Rainfall Prediction Using Machine Learning (Random Forest)

This project aims to predict rainfall using a supervised machine learning approach. Developed using Python in Google Colab, it utilizes the Random Forest Classifier along with essential data preprocessing, visualization, model evaluation, and model saving techniques.

## Project Overview

- Developed using Python in Google Colab
- Implements Random Forest Classifier from scikit-learn
- Performs data preprocessing, resampling, and feature analysis
- Includes visualizations such as heatmaps, boxplots, histograms, and class distributions
- Evaluates model using classification metrics and saves the trained model using pickle

## Technologies and Libraries Used

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn (RandomForestClassifier, train_test_split, GridSearchCV, etc.)
- pickle

## Exploratory Data Analysis (EDA)

### Correlation Heatmap

A heatmap was created to identify correlations between features and detect multicollinearity.

```python
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

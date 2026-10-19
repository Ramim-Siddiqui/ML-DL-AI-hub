# Fake News Detection with Data Visualization

This repository contains code for visualizing data and detecting fake news using machine learning techniques.

## Overview

The project aims to analyze and visualize numeric data and textual data related to news articles and develop a machine learning model to detect fake news. The repository includes code for:

- Data preprocessing and cleaning
- Exploratory data analysis (EDA) and visualization
- Feature extraction and selection
- Training and evaluating fake news detection models

## Notebooks and datasets

| Notebook | Dataset (in the same folder) | Models | Result |
|---|---|---|---|
| `Fake_News_Detection_System/Fake_News_Detection_System.ipynb` | `train.csv` - Kaggle "Fake News" competition (https://www.kaggle.com/c/fake-news/data), 20,800 articles with `id, title, author, text, label` | Porter stemming + TF-IDF on author + title, then Logistic Regression, Decision Tree, Gaussian Naive Bayes, Random Forest, Gradient Boosting; Embedding + LSTM on the full article text | Random Forest 99.4% test accuracy (best); LSTM 92.1% |
| `Data_Preprocessing_and_Machine_Learning/Data_Preprocessing_and_machine_learning_models.ipynb` | `pima-indians-diabetes.csv` - Pima Indians Diabetes (UCI), 768 rows | StandardScaler, then LDA, Naive Bayes, KNN, Decision Tree, Logistic Regression; repeated after downsampling the majority class | Best 74% (Naive Bayes / Logistic Regression); 78% after downsampling (LDA / Logistic Regression) |
| `Data_Visualization/Data_visualization.ipynb` | `housing.csv` - Boston Housing (UCI), 506 rows x 14 columns | Descriptive statistics, box plots, correlation matrix, scatter plots of the 5 features most correlated with price | EDA only |

The notebooks were written for Google Colab and read the data from Google Drive; change the `/content/drive/...` paths to the CSV files in each folder to run them locally.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK and WordCloud (fake-news notebook)
- TensorFlow < 2.16 (LSTM section; uses the legacy `keras.preprocessing.text` API)
- SciPy, statsmodels, wquantiles (visualization notebook)

Install them with pip:

```bash
pip install pandas numpy scikit-learn nltk wordcloud seaborn matplotlib scipy statsmodels wquantiles "tensorflow<2.16"
python -c "import nltk; nltk.download('stopwords')"
```

# Fake-News-Detection-Using-NLP-and-XAI
This project focuses on binary classification of news articles as real or fake, using Natural Language Processing (NLP) techniques and traditional machine learning models. It also integrates LIME (Local Interpretable Model-agnostic Explanations) to provide transparency and interpretability of model predictions.

Overview:
With the rise of misinformation, especially through online platforms, automated fake news detection has become a crucial task. This project leverages classical NLP techniques to clean and process textual data, builds models like Logistic Regression and Passive Aggressive Classifier, and incorporates LIME for model explainability.

Features:
Binary classification (FAKE vs REAL) using traditional ML algorithms
Data preprocessing with stopword removal, punctuation filtering, etc.
TF-IDF vectorization for text representation
Model evaluation using accuracy, confusion matrix, and classification report
Use of LIME to explain predictions for individual news articles

Dataset:
Source: [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset)
Two CSV files: Fake.csv and True.csv
Features used:
  title and text combined for input
  Custom binary labels created (FAKE = 0, REAL = 1)

Technologies:
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
LIME

Model Evaluation:
Accuracy: ~92% (Logistic Regression)
Passive Aggressive Classifier also showed competitive performance
Classification report used for detailed analysis

Explainability with LIME:
LIME helps visualize why a prediction was made, by highlighting which words in the article most contributed to the predicted label. This provides transparency, builds trust in the model, and helps identify patterns in misinformation.

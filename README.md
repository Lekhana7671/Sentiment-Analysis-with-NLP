# Sentiment-Analysis-with-NLP

*Company*: Codtech IT Solutions Private Limited

*Name*: Yarnakula Lekhana

*Intern ID*: CTIS1133

*Domain*: Machine Learning

*Duration*: 12 Weeks

*Mentor*: Neela Santosh

# 📝 Sentiment Analysis on Customer Reviews Using NLP

## 📌 Overview
This project demonstrates **Sentiment Analysis** on a dataset of customer reviews using **Natural Language Processing (NLP)** techniques. The task focuses on classifying customer feedback as **Positive** or **Negative** based on the textual content. 

The project utilizes **TF-IDF vectorization** to convert text into numerical features and a **Logistic Regression** classifier to predict sentiment. The workflow includes preprocessing, modeling, evaluation, and testing with new examples, all presented in a **Jupyter Notebook** compatible with Google Colab.

---

## 🎯 Objectives
- Perform sentiment classification of customer reviews using NLP.  
- Preprocess text data to clean and normalize input.  
- Convert text into numerical features using **TF-IDF vectorization**.  
- Train a **Logistic Regression** model for binary sentiment prediction.  
- Evaluate the model using accuracy, confusion matrix, and classification metrics.  
- Visualize results and interpret model predictions.  

---

## 📊 Dataset Description
For this internship task, a small **sample dataset** of customer reviews is used. Each review has an associated sentiment label:

| Feature | Description |
|---------|-------------|
| `review` | Text of the customer review |
| `sentiment` | Sentiment label: 1 = Positive, 0 = Negative |

The dataset contains a mixture of positive and negative reviews. Using a built-in or small sample dataset eliminates the need for downloading external files, making the implementation straightforward and reproducible.

---

## 🛠️ Tools & Technologies Used
- **Python 3.x**  
- **Google Colab / Jupyter Notebook**  
- **Pandas** – for data handling  
- **NumPy** – for numerical operations  
- **Scikit-learn** – for TF-IDF vectorization and Logistic Regression  
- **Regex (`re`)** – for text preprocessing  

---

## 🧪 Methodology

### 1. Data Preprocessing
- Convert all text to lowercase  
- Remove special characters and punctuation  
- Normalize text for better feature extraction  

### 2. Feature Extraction
- **TF-IDF Vectorization** transforms text into numerical feature vectors  
- Stopwords are removed to reduce noise  

### 3. Model Training
- **Logistic Regression** classifier is trained on TF-IDF features  
- Training set is separated from testing set using **train-test split**  

### 4. Prediction & Evaluation
- Model predicts sentiment on unseen test data  
- Evaluation metrics include:
  - **Accuracy**  
  - **Classification Report** (Precision, Recall, F1-score)  
  - **Confusion Matrix**  

### 5. Testing with New Reviews
- Model can predict sentiment for new customer reviews  
- Demonstrates real-world application of the trained model  

---

## 📈 Results & Analysis
The Logistic Regression model combined with TF-IDF vectorization successfully classifies customer reviews as **Positive** or **Negative**. Text preprocessing improves accuracy by reducing noise and irrelevant information. The confusion matrix and classification report indicate how well the model predicts each class. The notebook also provides the capability to test new reviews, showing the practical applicability of the model.

---

## ✅ Deliverables
- `Sentiment_Analysis_NLP.ipynb` – Jupyter Notebook containing:
  - Data preprocessing  
  - TF-IDF vectorization  
  - Logistic Regression model  
  - Model evaluation metrics  
  - Predictions on new reviews  
  - Analysis and observations  

---

## 🙌 Acknowledgments
This project was completed as part of an internship task at **CODTECH**. It demonstrates fundamental NLP techniques for sentiment analysis and practical machine learning implementation for customer review classification.

---

## 🚀 How to Run
1. Open the notebook in **Google Colab** or **Jupyter Notebook**  
2. Run all cells sequentially (Shift + Enter)  
3. Explore model predictions and evaluation metrics  

---

## 📌 Notes
- Small sample dataset is sufficient for demonstrating workflow  
- TF-IDF and Logistic Regression are widely used for text classification  
- Notebook is reproducible and can be extended to larger datasets  

## 📸 Output

<img width="1913" height="867" alt="Image" src="https://github.com/user-attachments/assets/aca86080-9b66-4dd9-ba58-443e429340a1" />

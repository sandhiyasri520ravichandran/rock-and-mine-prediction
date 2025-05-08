                                                                         🎯 Rock vs Mine Prediction using SONAR Data

This project demonstrates how to build a binary classification system in Python that predicts whether an object is a rock or a mine using SONAR dataset readings. The model is built using Logistic Regression and implemented in Google Colaboratory for ease of experimentation and reproducibility.

📂 Dataset
We use the SONAR dataset which contains 60 numerical attributes representing sonar signal energy bounced off an object.

Each instance is labeled as either 'R' (rock) or 'M' (mine).

 Tools & Technologies

•	 Python
•	Scikit-learn
•	 NumPy, Pandas
•	 Logistic Regression
•	Google Colaboratory

How to Run
Open the notebook in Google Colab
Upload your dataset or mount Google Drive if needed.

Install required libraries
Most required libraries are pre-installed in Colab. 

Run:
!pip install -q numpy pandas matplotlib seaborn scikit-learn
Load and preprocess the data

import pandas as pd
data = pd.read_csv('/path/to/sonar.csv', header=None)

Split dataset & train model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

📈 Results
The logistic regression model generally achieves an accuracy between 75% - 85%, depending on the train-test split and random seed.

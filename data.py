import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import sklearn

# Load dataset
df = pd.read_csv("./data/Admission_Predict.csv")

print("Pandas: "+pd.__version__)  # 2.2.3
print("numpy: "+np.__version__) # 2.2.2
print("sklearn: "+sklearn.__version__) # 1.6.1

# Data Cleaning: Drop Serial No.
df_cleaned = df.drop(columns=["Serial No."])

# Convert 'Chance of Admit' into a binary classification (Threshold: 0.75)
df_cleaned['Admitted'] = (df_cleaned['Chance of Admit '] >= 0.75).astype(int)
df_cleaned = df_cleaned.drop(columns=['Chance of Admit '])

# Define features and target
X = df_cleaned.drop(columns=['Admitted'])
y = df_cleaned['Admitted']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")

# Overfitting and underfitting check using learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring='accuracy')
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve to Detect Overfitting/Underfitting")
plt.legend()
plt.show()

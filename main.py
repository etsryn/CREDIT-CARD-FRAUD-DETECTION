# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Step 1: Load the Credit Card Fraud Dataset
print("Step 1: Loading the Credit Card Fraud Dataset...")
data = pd.read_csv('credit_card_data.csv')

# Step 2: Handle Missing Values
print("Step 2: Handling Missing Values...")
data.dropna(subset=['Class'], inplace=True)

# Step 3: Data Preparation
print("Step 3: Data Preparation...")
X = data.drop(columns=['Class'])
y = data['Class']

# Step 4: Splitting the Dataset
print("Step 4: Splitting the Dataset into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Initialize and Train the Classifier
print("Step 5: Initializing and Training the Random Forest Classifier...")
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make Predictions on the Test Set
print("Step 6: Making Predictions on the Test Set...")
y_pred = clf.predict(X_test)

# Step 7: Model Evaluation
print("Step 7: Evaluating the Model...")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Step 8: Visualize the Confusion Matrix
print("Step 8: Visualizing the Confusion Matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='viridis', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - Fraud Detection')
plt.show()

# Plot ROC curve
print("Plotting ROC curve...")
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of class 1 (fraudulent)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Step 10: Conclusion
print("Step 10: Conclusion")
print("The random forest classifier has been trained and evaluated for credit card fraud detection.")
print("The accuracy, confusion matrix, classification report, and ROC curve have been analyzed to assess the model's performance.")

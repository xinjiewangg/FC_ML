import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import learning_curve

#Load the dataset
df = pd.read_csv("data/minerals/minerals_36k.csv")

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, random_state=10)

# Input the best hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=2)


# Saving the model
import joblib 
# # replace the name 
# # joblib.dump(rf_classifier, 'trained_rf_classifier_3BMPs.pkl') 
# # joblib.dump(scaler, 'scaler.pkl')
# # print("Trained model and scaler saved.")

# Check if a trained model file exists and load it, if available
try:
    rf_classifier = joblib.load('trained_rf_minerals_36k.pkl')
    print("Trained model loaded successfully.")
except FileNotFoundError:
    # Fit the model to the training data if the saved model file is not found
    rf_classifier.fit(X_train, y_train)
    # Save the trained classifier to a file
    joblib.dump(rf_classifier, 'trained_rf_minerals_36k.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Trained model and scaler saved.")
# ######




cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Predict labels for the training dataset
predicted_train = rf_classifier.predict(X_train)

# Calculate accuracy, precision, recall for training dataset
accuracy_train = accuracy_score(y_train, predicted_train)
precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(y_train, predicted_train, average='weighted')

# Calculate specificity for each class in the training dataset
specificity_train = {}

for label in np.unique(labels):
    binary_true_labels = np.where(y_train == label, 1, 0)
    binary_predicted_labels = np.where(predicted_train == label, 1, 0)
    cm_class = confusion_matrix(binary_true_labels, binary_predicted_labels)
    
    TN_class = cm_class[0, 0]
    FP_class = cm_class[0, 1]
    
    specificity_class = TN_class / (TN_class + FP_class)
    specificity_train[f'Specificity (Class {label})'] = specificity_class

# Print classification report for the training dataset with specificity
report_train = classification_report(y_train, predicted_train, target_names=[f'Class {i}' for i in np.unique(labels)])
print("Classification Report (Training Data):")
print(report_train)
print(f"Accuracy (Training Data): {accuracy_train:.2f}")
print(f"Precision (Training Data): {precision_train:.2f}")
print(f"Recall (Training Data): {recall_train:.2f}")

# Print specificity for each class in the training dataset
for class_label, specificity in specificity_train.items():
    print(f"{class_label}: {specificity:.2f}")

# Predict labels for the test dataset
predicted_test = rf_classifier.predict(X_test)

# Calculate accuracy, precision, recall for test dataset
accuracy_test = accuracy_score(y_test, predicted_test)
precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(y_test, predicted_test, average='weighted')


# Calculate specificity for each class in the test dataset
specificity_test = {}

for label in np.unique(labels):
    binary_true_labels = np.where(y_test == label, 1, 0)
    binary_predicted_labels = np.where(predicted_test == label, 1, 0)
    cm_class = confusion_matrix(binary_true_labels, binary_predicted_labels)
    
    TN_class = cm_class[0, 0]
    FP_class = cm_class[0, 1]
    
    specificity_class = TN_class / (TN_class + FP_class)
    specificity_test[f'Specificity (Class {label})'] = specificity_class


# Print classification report for the test dataset with specificity
report_test = classification_report(y_test, predicted_test, target_names=[f'Class {i}' for i in np.unique(labels)])
print("\nClassification Report (Test Data):")
print(report_test)
print(f"Accuracy (Test Data): {accuracy_test:.2f}")
print(f"Precision (Test Data): {precision_test:.2f}")
print(f"Recall (Test Data): {recall_test:.2f}")

# Print specificity for each class in the test dataset
for class_label, specificity in specificity_test.items():
    print(f"{class_label}: {specificity:.2f}")

# Calculate precision, recall, F1-score, and support for each class in the test dataset
precision, _, _, _ = precision_recall_fscore_support(y_test, predicted_test, labels=np.unique(labels), average=None)

# Print accuracy for each class
for i, class_label in enumerate(np.unique(labels)):
    print(f"Accuracy (Class {class_label}): {precision[i]:.2f}")



# Plot confusion matrix for the test dataset
cm_test = confusion_matrix(y_test, predicted_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.unique(labels))
disp_test.plot(cmap=plt.cm.BuGn)
disp_test.ax_.xaxis.label.set_fontsize(14)  
disp_test.ax_.yaxis.label.set_fontsize(14)  
disp_test.ax_.xaxis.get_label().set_fontsize(14)  
disp_test.ax_.yaxis.get_label().set_fontsize(14)  
disp_test.im_.colorbar.ax.tick_params(labelsize=14)

disp_test.im_.axes.set_xticklabels(disp_test.im_.axes.get_xticklabels(), fontsize=14)  
disp_test.im_.axes.set_yticklabels(disp_test.im_.axes.get_yticklabels(), fontsize=14) 

plt.title('Confusion Matrix (Test Data)', fontsize=14)
plt.savefig("Confusion Matrix_Test Data.png", dpi=300)
plt.show()


# Plot normalized confusion matrix for the test dataset
cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
disp_test_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_test_normalized, display_labels=np.unique(labels))
disp_test_normalized.plot(cmap=plt.cm.BuGn, values_format='.2f', ax=plt.gca())  
disp_test_normalized.ax_.xaxis.label.set_fontsize(14)  
disp_test_normalized.ax_.yaxis.label.set_fontsize(14)  
disp_test_normalized.ax_.xaxis.get_label().set_fontsize(14)  
disp_test_normalized.ax_.yaxis.get_label().set_fontsize(14)  
disp_test_normalized.im_.colorbar.ax.tick_params(labelsize=14)

disp_test_normalized.im_.axes.set_xticklabels(disp_test.im_.axes.get_xticklabels(), fontsize=14)  
disp_test_normalized.im_.axes.set_yticklabels(disp_test.im_.axes.get_yticklabels(), fontsize=14)  

plt.title('Normalized Confusion Matrix (Test Data)', fontsize=14)  
plt.savefig("Normalized Confusion Matrix_Test Data.png", dpi=300)
plt.show()

# Learning curve
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


from sklearn.inspection import permutation_importance
feature_importances = rf_classifier.feature_importances_
feature_names = features.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n = 12
print(f"Top {top_n} Most Important Features:")
print(feature_importance_df.head(top_n))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), feature_importance_df['Importance'][:top_n], align='center',  color='mediumaquamarine')
plt.yticks(range(top_n), feature_importance_df['Feature'][:top_n])

plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Feature Name', fontsize=14)
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
# plt.gca().invert_yaxis()  

plt.savefig("feature_importance_plot.png", dpi=300)  
plt.show()  



# Create a heatmap of feature importances
import seaborn as sns
import matplotlib.pyplot as plt

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select the top N most important features
top_n = 12
top_features = feature_importance_df.head(top_n)

top_features = top_features[::-1]

# plot a heatmap of feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='YlGn')
#plt.title('Top Feature Importances (Random Forest)', fontsize=16)
plt.xlabel('Importance', fontsize=22)
plt.ylabel('Feature Name', fontsize=22)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  

plt.tight_layout()
plt.savefig("feature_importance_heatmap.png", dpi=300)
plt.show()

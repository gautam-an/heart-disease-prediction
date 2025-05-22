import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('heart.csv')

print("\ndataset info:")
print("shape (rows, columns):", df.shape)
print("\ntarget distribution:")
target_count = df['target'].value_counts()
print(f"non disease cases: {target_count[0]}")
print(f"disease cases: {target_count[1]}")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
train = scaler.fit_transform(X_train)  # scaled
test= scaler.transform(X_test)         # scaled

print("\ntraining random forest model...")
model = RandomForestClassifier(random_state=42)
model.fit(train, y_train)

y_pred = model.predict(test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\nresults:")
print(f"acc: {accuracy:.2f}%")

print("")
more_results = input("do you want to see more results? (y/n): ")
if more_results.lower() == 'y':
    print("\nclassification report:")
    print(classification_report(y_test, y_pred))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nperforming hyperparameter tuning...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(train, y_train)

print("\nbest params:", grid_search.best_params_)
score = grid_search.best_score_ * 100  
print(f"best cross validation score: {score:.2f}%")  

tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(test)

accuracy_tuned = accuracy_score(y_test, y_pred_tuned) * 100

print("\ntuned model Results:")
print(f"acc: {accuracy_tuned:.2f}%")

print("")
more_results1 = input("do you want to see more results? (y/n): ")
if more_results1.lower() == 'y':
    print("\nclassification report:")
    print(classification_report(y_test, y_pred_tuned))
    print("confusion matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))

def plot_feature_importance(model, feature_names, filename='feature_importance.png'):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 8))
    
    plt.barh(range(len(indices)), importances[indices], align='center')
    
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Heart Disease Prediction')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"\nfeature importance plot saved as {filename}")

plot_feature_importance(tuned_model, X.columns)

joblib.dump(tuned_model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

def predict(patient_data):

    patient_df = pd.DataFrame([patient_data])
    patient_scaled = scaler.transform(patient_df)
    
    prediction = tuned_model.predict(patient_scaled)[0]
    probability = tuned_model.predict_proba(patient_scaled)[0][1]  
    
    return prediction, probability

print("\nresults for example prediction:")
example_patient = {
    'age': 55,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 240,
    'fbs': 0,
    'restecg': 1,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 1,
    'thal': 2
}

prediction, probability = predict(example_patient)
print(f"prediction: {'heart disease' if prediction == 1 else 'no heart disease'}")
probability = probability * 100
print(f"probability of getting heart disease: {probability:.2f}%")

print("\nmodel and scalar saved in heart_disease_model.pkl and scaler.pkl")
print("")

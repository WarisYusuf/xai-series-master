# %% Imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import shap
import time
from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)
# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
# https://scikit-learn.org/stable/modules/svm.html
clf = make_pipeline( StandardScaler(),
    SVC(probability=True, random_state=2021))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:, 1]

print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Average precision:", average_precision_score(y_test, y_prob))

print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")


d = shap.sample(X_train, random_state=2021,nsamples= 50)

explainer = shap.KernelExplainer(lambda a: clf.predict_proba(a), d)

X_sample = X_test.iloc[:100] # only 100 patiens selected to test

# Calculate SHAP values for the selected patients
start = time.perf_counter()

shap_values = explainer.shap_values(X_sample, nsamples=100)

runtime = time.perf_counter() - start

print("SHAP runtime (seconds):", runtime)
print("SHAP values shape:", np.array(shap_values).shape)

# Check reconstruction for the selected patient
patient_index = 1
stroke_index = 1

base_value = explainer.expected_value[stroke_index]
shap_sum = np.sum(shap_values[patient_index, :, stroke_index])
reconstructed = base_value + shap_sum

predicted = clf.predict_proba(
    X_sample.iloc[[patient_index]]
)[0, stroke_index]

print("Selected class:", clf.classes_[stroke_index])
print("Reconstructed probability:", reconstructed)
print("Predicted probability:", predicted)
print("Reconstruction residual:", abs(reconstructed - predicted))

# %% >> Visualize local predictions
shap.initjs()
patient_index = 1

# Force plot
prediction = clf.predict(X_sample.iloc[[patient_index]])[0]
print(f"The SVM predicted: {prediction}")

a = shap.force_plot(explainer.expected_value[1],
                shap_values[patient_index, :, stroke_index],
                X_sample.iloc[patient_index]) # for values
shap.save_html("svm_plot.html", a)



shap.summary_plot(shap_values[:, :, 1], # create for shap values for stroke 1
    X_sample,
    show=False
)


plt.savefig("svm.png")

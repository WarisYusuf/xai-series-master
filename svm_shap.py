# %% Imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap
import matplotlib.pyplot as plt
import numpy as np

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
clf = make_pipeline(StandardScaler(), SVC(probability=True))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")


d = shap.sample(X_train, random_state=2021,nsamples= 50)

explainer = shap.KernelExplainer(lambda a: clf.predict_proba(a), d)

X_sample = X_test.iloc[:100] # only 100 patiens selected to test

# Calculate SHAP values for the selected patients
shap_values = explainer.shap_values(X_sample, nsamples=100)

print("SHAP values shape:", np.array(shap_values).shape)


shap.summary_plot(shap_values[:, :, 1], # create for shap values for stroke 1
    X_sample,
    show=False
)


plt.savefig("svm.png")

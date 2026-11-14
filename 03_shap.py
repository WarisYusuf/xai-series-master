# %% Imports
from utils import DataLoader
import sklearn
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

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

rf = RandomForestClassifier(random_state=2021)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
stroke_index = list(rf.classes_).index(1)
y_prob = rf.predict_proba(X_test)[:, stroke_index]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
print("Precision (stroke=1):", precision_score(y_test, y_pred, zero_division=0))
print("Recall (stroke=1):", recall_score(y_test, y_pred, zero_division=0))
print("F1 (stroke=1):", f1_score(y_test, y_pred, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Average precision:", average_precision_score(y_test, y_prob))

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
print("Positive class support:", (y_test == 1).sum())
print("Train size after oversampling:", len(X_train))
print("Test size:", len(X_test))
print("Classes:", rf.classes_)

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]

# %% Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print(shap_values[0].shape)
shap_values

# %% >> Visualize local predictions
shap.initjs()
# Force plot
prediction = rf.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")
patient_index = 0
stroke_index = 1
a = shap.force_plot(explainer.expected_value[1],
                shap_values[patient_index, :, stroke_index],
                X_test[start_index:end_index]) # for values
shap.save_html("plot.html", a)

print("Actual class:", y_test.iloc[1])

# %% >> Visualize global features
# Feature summary
#shap.summary_plot(shap_values, X_test)

shap_values_all = explainer.shap_values(X_test)
shap.summary_plot(shap_values_all[:, :, 1], X_test)



# %% >> Visualize global features

shap_values_all = explainer.shap_values(X_test)

# Create and save the SHAP summary plot
shap.summary_plot(
    shap_values_all[:, :, 1],
    X_test,
    show=False
)

plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# Calculate the five most important features
global_importance = np.abs(
    shap_values_all[:, :, 1]
).mean(axis=0)

feature_ranking = sorted(
    zip(X_test.columns, global_importance),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 5 Global SHAP Features:")

for feature, importance in feature_ranking[:5]:
    print(feature, ":", round(importance, 6))

print("Test samples:", len(X_test))
print("Positive test cases:", (y_test == 1).sum())
print("Number of features:", X_test.shape[1])


print("SHAP version:", shap.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Model classes:", rf.classes_)
print("Feature count:", X_test.shape[1])
print("Test sample size:", len(X_test))
print("Local explanation size:", end_index - start_index)

print("Explainer model output:", explainer.model.model_output)
print("Expected values:", explainer.expected_value)

# Verify SHAP contributions for Class 1
case = X_test.iloc[start_index:start_index + 1]
values = explainer.shap_values(case)

base_value = explainer.expected_value[1]
shap_sum = base_value + values[0, :, 1].sum()
predicted_probability = rf.predict_proba(case)[0, 1]

print("Base value + SHAP:", shap_sum)
print("Predicted probability:", predicted_probability)
print("Difference:", abs(shap_sum - predicted_probability))
print("Feature names match:", list(X_test.columns) == list(rf.feature_names_in_))


# %% Section 4 - Local explanation details

case = X_test.iloc[1:2]
actual = y_test.iloc[1]
predicted = rf.predict(case)[0]
probability = rf.predict_proba(case)[0, 1]

local_shap = explainer.shap_values(case)[0, :, 1]

print("Actual class:", actual)
print("Predicted class:", predicted)
print("Stroke probability:", probability)
print("Base value:", explainer.expected_value[1])
print("Original row index:", X_test.index[1])

contributions = sorted(
    zip(case.columns, case.iloc[0], local_shap),
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\nTop SHAP contributions:")
for feature, value, contribution in contributions[:8]:
    print(feature, "| Value:", value,
          "| SHAP:", round(contribution, 6))

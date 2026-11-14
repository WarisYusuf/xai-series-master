# %% Imports
import matplotlib.pyplot as plt
from utils import DataLoader

# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print(data.shape)
print(data["stroke"].value_counts())
print(data["stroke"].value_counts(normalize=True) * 100)
print("Duplicate records:", data.duplicated().sum())
print("Duplicate IDs:", data["id"].duplicated().sum())
print("Missing values:")
print(data.isnull().sum())
print("Missing values in %:")
print((data.isnull().mean() * 100).round(2))
print("Unknown smoking status:")
print((data["smoking_status"] == "Unknown").sum())
print(data[["age", "avg_glucose_level", "bmi"]].describe().round(2))
print("\nCategorical distributions")
categorical_columns = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical_columns:
    print(f"\n{col}:")
    counts = data[col].value_counts()
    percentages = data[col].value_counts(normalize=True) * 100

    for category in counts.index:
        print(f"{category}: {counts[category]} ({percentages[category]:.2f}%)")
data.head()

# %% Show general statistics
data.info()
# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()



fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# 1. Stroke class counts
data["stroke"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0, 0]
)
axes[0, 0].set_title("Stroke Class Distribution")
axes[0, 0].set_xlabel("Stroke (0 = No, 1 = Yes)")
axes[0, 0].set_ylabel("Number of patients")

# 2. Age distribution
data["age"].hist(bins=30, ax=axes[0, 1])
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age (years)")
axes[0, 1].set_ylabel("Number of patients")

# 3. Glucose distribution
data["avg_glucose_level"].hist(bins=30, ax=axes[0, 2])
axes[0, 2].set_title("Average Glucose Distribution")
axes[0, 2].set_xlabel("Average glucose level")
axes[0, 2].set_ylabel("Number of patients")

# 4. BMI distribution
data["bmi"].hist(bins=30, ax=axes[1, 0])
axes[1, 0].set_title("BMI Distribution")
axes[1, 0].set_xlabel("BMI")
axes[1, 0].set_ylabel("Number of patients")

# 5. Missing values
data.isnull().sum().plot(kind="bar", ax=axes[1, 1])
axes[1, 1].set_title("Missing Values")
axes[1, 1].set_xlabel("Variables")
axes[1, 1].set_ylabel("Number of missing values")
axes[1, 1].tick_params(axis="x", rotation=90)

# 6. Age comparison by stroke status
data.boxplot(column="age", by="stroke", ax=axes[1, 2])
axes[1, 2].set_title("Age by Stroke Status")
axes[1, 2].set_xlabel("Stroke (0 = No, 1 = Yes)")
axes[1, 2].set_ylabel("Age (years)")

plt.suptitle("Exploratory Data Analysis", fontsize=15)
plt.savefig("eda_panel.png")
plt.show()


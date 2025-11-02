# penguins_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# === 1. Load Dataset ===
data_path = "penguins_size.csv"  # make sure the file is in the same folder as this script
data = pd.read_csv(data_path)

print("=== Dataset Preview ===")
print(data.head(), "\n")

# === 2. Data Cleaning ===
# Drop rows with missing or invalid 'sex' values
data = data.dropna(subset=["sex"])
data = data[data["sex"].isin(["MALE", "FEMALE"])]

# Drop rows with missing numerical values
data = data.dropna()

# === 3. Encode Categorical Columns ===
label_encoders = {}
for col in ["species", "island", "sex"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Separate features (X) and target (y)
X = data.drop(columns=["sex"])
y = data["sex"]

# === 4. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# === 5. Train the Model ===
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# === 6. Predictions ===
y_pred = clf.predict(X_test)

# === 7. Decode labels safely ===
# Convert numeric back to readable labels (male/female)
sex_encoder = label_encoders["sex"]
y_test_decoded = sex_encoder.inverse_transform(y_test)
y_pred_decoded = sex_encoder.inverse_transform(y_pred)

# === 8. Model Evaluation ===
print("=== Model Evaluation ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=sex_encoder.classes_), "\n")

# === 9. Actual vs Predicted ===
results_df = pd.DataFrame({"Actual": y_test_decoded, "Predicted": y_pred_decoded})
print("=== Actual vs Predicted (Sample) ===")
print(results_df.head(), "\n")

# === 10. Feature Importances ===
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("=== Feature Importance (Descending) ===")
print(feature_importances, "\n")

# === 11. Visualize the Decision Tree ===
plt.figure(figsize=(14, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=sex_encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Penguin Gender Classification")
plt.show()

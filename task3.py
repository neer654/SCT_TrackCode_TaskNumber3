import zipfile
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load CSV from nested ZIP

# Path to the outer ZIP file
outer_zip_path = r"D:\downloads\bank+marketing.zip"

# Open the outer ZIP
with zipfile.ZipFile(outer_zip_path) as outer_zip:
    # Open inner ZIP file inside the outer ZIP
    with outer_zip.open("bank-additional.zip") as inner_zip_file:
        inner_zip_bytes = inner_zip_file.read()

        # Load the inner ZIP from memory
        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
            # List files inside
            print("Files inside inner ZIP:", inner_zip.namelist())

            # Load the CSV file from the inner ZIP
            with inner_zip.open("bank-additional/bank-additional-full.csv") as csv_file:
                df = pd.read_csv(csv_file, sep=';')

# Data Preprocessing

# Print dataset info
print("Shape of dataset:", df.shape)
print("Target variable value counts:\n", df['y'].value_counts())

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode target column separately
target_le = label_encoders['y']

# Split Data
X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Model
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualize Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=target_le.classes_, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

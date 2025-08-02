Bank Marketing Decision Tree Classifier

Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

This project uses the [Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI Machine Learning Repository to train a supervised machine learning model that classifies customers as likely or unlikely to subscribe to a term deposit.

📌 Objective

Develop a machine learning pipeline using a Decision Tree to:
- Load and preprocess customer data from a nested ZIP archive
- Encode categorical features
- Train a classifier to predict subscription outcome
- Evaluate performance using classification metrics
- Visualize the decision-making process

⚙️ Technologies Used

- Python
- pandas & NumPy
- scikit-learn
- matplotlib & seaborn
- zipfile & io (for nested ZIP handling)

📂 Dataset

- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- File: `bank-additional-full.csv` (nested inside ZIP archives)
- Records: 41,188
- Features: 20 input attributes (age, job, education, contact type, etc.)
- Target: `y` (binary — `yes` or `no`)

📈 Output

- Classification Report with Precision, Recall, F1-Score
- Accuracy Score
- Confusion Matrix (visualized)
- Decision Tree Visualization

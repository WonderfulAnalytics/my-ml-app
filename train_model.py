# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
print("Loading dataset...")
df = pd.read_csv('insurance.csv')

# Preprocess dataset
print("Preprocessing dataset...")
df = pd.get_dummies(df, drop_first=True)

# Split dataset
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and columns
print("Saving model and columns...")
joblib.dump(model, 'model.pkl')
joblib.dump(X_train.columns, 'model_columns.pkl')
print("Model and columns saved successfully!")

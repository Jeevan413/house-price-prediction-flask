import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load data
data = pd.read_csv("house_data.csv")

# 2. Features and target
X = data[['Size', 'Bedrooms', 'Age']]
y = data['Price']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Save model
joblib.dump(model, "house_price_model.pkl")

print("Model trained and saved as house_price_model.pkl")

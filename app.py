import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

df = pd.DataFrame({
    'Age': np.random.normal(40, 12, 1000),
    'Monthly_Charges': np.random.normal(70, 25, 1000),
    'Churn': np.random.randint(0, 2, 1000)
})

X = df[['Age', 'Monthly_Charges']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
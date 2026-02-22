
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Loading Data...")
sorvetes = pd.read_csv('Vendas de Sorvete.csv', sep=';')  # ← fix aqui

print("Colunas:", sorvetes.columns.tolist())
print(sorvetes.head())

X = sorvetes[['Temperatura (°C)']].values
y = sorvetes['Vendas (unidades)'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

mlflow.autolog()

model = LinearRegression()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_hat))
r2   = r2_score(y_test, y_hat)

print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

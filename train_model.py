import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Carregar o dataset processado
df_encoded = pd.read_csv('data/processed_insurance.csv')

# Dividir o dataset em conjuntos de treinamento e teste
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Seleção do modelo
model = LinearRegression()

# Treinamento do modelo
model.fit(X_train, y_train)

# Validação Cruzada
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Pontuações de RMSE: {rmse_scores}")
print(f"RMSE Médio: {rmse_scores.mean()}")

# Salvar o modelo treinado
with open('data/medical_cost_model.pkl', 'wb') as file:
    pickle.dump(model, file)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

# Carregar o dataset processado
df_encoded = pd.read_csv('data/processed_insurance.csv')

# Dividir o dataset em conjuntos de treinamento e teste
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Carregar o modelo treinado
with open('data/medical_cost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Cálculo das métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plotando resíduos
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Distributição de Resíduos')
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
df = pd.read_csv('data/insurance.csv')

# Verificação de valores ausentes
print(df.isnull().sum())

# Remover todas as linhas que possuem valores missing
df.dropna(inplace=True)


# One-Hot Encoding para variáveis categóricas
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Normalizar as variáveis numéricas
scaler = StandardScaler()
numerical_columns = df.select_dtypes(include=[np.number]).columns
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Salvar o dataframe processado
df_encoded.to_csv('data/processed_insurance.csv', index=False)
print(df_encoded.head())

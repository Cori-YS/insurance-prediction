import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de visualização
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Carregar o dataset
df = pd.read_csv('data/insurance.csv')

# Exibir as primeiras linhas do dataset
print("Primeiras linhas do dataset:")
print(df.head())

# Exibir estatísticas descritivas
print("\nEstatísticas Descritivas:")
print(df.describe())

# Selecionar apenas colunas numéricas
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Calcular e exibir a mediana para cada coluna numérica
print("\nMedia para cada coluna numérica:")
print(df[numerical_columns].mean())

# Calcular e exibir a mediana para cada coluna numérica
print("\nMediana para cada coluna numérica:")
print(df[numerical_columns].median())

# Calcular e exibir o desvio padrão para cada coluna numérica
print("\nDesvio Padrão para cada coluna numérica:")
print(df[numerical_columns].std())

# Verificar tipos de dados e valores ausentes
print("\nInformações sobre os dados:")
print(df.info())

# Análise de Correlação
print("\nMatriz de Correlação:")
corr = df[numerical_columns].corr()
print(corr)

# Heatmap da Matriz de Correlação
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap da Matriz de Correlação')
plt.show()

# Histogramas para distribuição das variáveis numéricas
histogram = df[numerical_columns].hist(bins=30, edgecolor='black')
plt.suptitle('Histogramas para distribuição das variáveis numéricas', y=1.02)
plt.tight_layout()
plt.show()

# Box plots para detectar outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title('Box plots das variáveis numéricas')
plt.show()

# Scatter plot para relações entre variáveis
scatterplot = sns.pairplot(df, height=2)
scatterplot.fig.suptitle('Scatter plot para relações entre variáveis', y=1.02)
plt.tight_layout()
plt.show()

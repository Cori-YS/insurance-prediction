import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Carregar o modelo
with open('data/medical_cost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Função para desnormalizar a previsão
def denormalize_cost(normalized_cost):
    # Carregar o dataset original
    df = pd.read_csv('data/insurance.csv')
    std = df['charges'].std()
    mean = df['charges'].mean()
    return normalized_cost * std + mean

# Função para previsão
def predict_cost(age, sex, bmi, children, smoker, region):
    # Criação do dataframe com os dados de entrada
    data = {
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_male': [1 if sex == 'male' else 0],
        'smoker_yes': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    }
    input_df = pd.DataFrame(data)
    # Normalizar as variáveis numéricas
    scaler = StandardScaler()
    numerical_columns = ['age', 'bmi', 'children']
    input_df[numerical_columns] = scaler.fit_transform(input_df[numerical_columns])

    prediction = model.predict(input_df)
    return denormalize_cost(prediction[0])
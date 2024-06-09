# Instale o streamlit via pip: pip install streamlit
import requests
import streamlit as st

# Interface do usuário com Streamlit
st.title('Predição de Custos Médicos')
age = st.number_input('Idade', min_value=18, max_value=100, value=25)
sex = st.selectbox('Sexo', ['male', 'female'])
bmi = st.number_input('BMI', min_value=15.0, max_value=40.0, value=22.0)
children = st.number_input('Número de Filhos', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Fumante', ['yes', 'no'])
region = st.selectbox('Região', ['northeast', 'northwest', 'southeast', 'southwest'])

if st.button('Calcular Custo'):
    # URL da API Flask
    api_url = 'http://localhost:5000/predict'
    
    # Dados para a previsão
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Enviar solicitação POST para a API
    response = requests.post(api_url, json=data)
    
    # Obter a previsão do JSON de resposta
    prediction = response.json()['prediction']
    
    st.write(f'O custo médico predito é {prediction:.2f} AOA')
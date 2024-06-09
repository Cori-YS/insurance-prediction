# Predição de Custos Médicos

## Membros do Grupo

- ISMÉLIO CORI
- ALBERTO CARDOSO
- ALDAIR LUENDA
- MANUEL NGOMBO
- PEDRO GONÇALVES

## Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo preditivo para estimar os custos médicos individuais com base em características demográficas e de saúde dos beneficiários.

## Requisitos

Certifique-se de ter o Python instalado e as seguintes bibliotecas:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- flask
- requests

Você pode instalar as bibliotecas necessárias com o seguinte comando:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit flask requests
```

## Executando o Projeto

### 1. Efetuar Análise Exploratória

Para realizar a análise exploratória dos dados, execute:

```bash
python eda.py
```

### 2. Efetuar Pré-processamento

Para realizar o pré-processamento dos dados, execute:

```bash
python preprocess.py
```

### 3. Efetuar Treinamento do Modelo

Para treinar o modelo, execute:

```bash
python train_model.py
```

### 4. Efetuar Avaliação do Modelo

Para avaliar o modelo, execute:

```bash
python evaluate_model.py
```

### 5. Iniciar API

Para iniciar a API Flask, execute:

```bash
python api.py
```

### 6. Iniciar App com Interface Front End

Para iniciar a interface do usuário com Streamlit, execute:

```bash
streamlit run app.py
```

## Estrutura do Projeto

```
project_directory/
├── predict.py
├── eda.py
├── preprocess.py
├── train_model.py
├── evaluate_model.py
├── api.py
├── app.py
├── data/medical_cost_model.pkl
├── data/insurance.csv
├── data/processed_insurance.csv
└── README.md
```

## Descrição dos Arquivos

- `predict.py`: Contém funções utilitárias como `predict_cost`.
- `eda.py`: Realiza a análise exploratória dos dados.
- `preprocess.py`: Realiza o pré-processamento dos dados.
- `train_model.py`: Treina o modelo de predição.
- `evaluate_model.py`: Avalia o modelo treinado.
- `api.py`: Implementa a API Flask para realizar previsões.
- `app.py`: Interface de usuário em Streamlit para entrada de dados e visualização das previsões.
- `data/medical_cost_model.pkl`: Arquivo do modelo treinado.
- `data/insurance.csv`: Dataset utilizado para o projeto.
- `data/processed_insurance.csv`: Dataset processado no projeto.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir um problema ou enviar um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

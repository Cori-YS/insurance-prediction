# Instale o Flask via pip: pip install flask
from flask import Flask, request, jsonify
from predict import predict_cost

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data['age']
    sex = data['sex']
    bmi = data['bmi']
    children = data['children']
    smoker = data['smoker']
    region = data['region']
    
    prediction = predict_cost(age, sex, bmi, children, smoker, region)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=False)
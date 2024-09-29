from flask import Flask, request, jsonify
import joblib
import pandas as pd

model = joblib.load('fire_prediction_model.pkl')

app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    return jsonify({'fire_prediction': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

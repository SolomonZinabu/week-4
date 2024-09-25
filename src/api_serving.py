# api_serving.py
from flask import Flask, request, jsonify
import joblib

def start_flask_api():
    app = Flask(__name__)

    # Load the trained model
    model = joblib.load('model.pkl')

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        prediction = model.predict([data['features']])
        return jsonify({'prediction': prediction.tolist()})

    app.run(debug=True)


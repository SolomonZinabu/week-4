from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction = model.predict([input_data])
    return jsonify({'sales_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

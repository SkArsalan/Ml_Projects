from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
from flasgger import Swagger
from flasgger.utils import swag_from

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)

model, count_vector = joblib.load('spam_detect_model.pkl')

@app.route("/")
@app.route("/home")
@swag_from('swagger/swagger_home.yaml', methods=['GET'])
def home():
    return "<h1>This is Home Page</h1>"

@app.route('/predict', methods=['POST'])
@swag_from('swagger/swagger_predict.yaml', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    vectorized_text = count_vector.transform([text])
    prediction = model.predict(vectorized_text)
    return jsonify({'prediction':int(prediction[0])})
if __name__ == "__main__":
    app.run(port=5002, debug=True)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Load ML models
try:
    label_encoder = joblib.load('label_encoder.pkl')
    ohe_encoder = joblib.load('ohe_encoder.pkl')
    model = joblib.load('lightgbm_crop_model.pkl')
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not all(field in data for field in ['State', 'Season', 'Annual_Rainfall']):
            return jsonify({'error': 'Missing required fields'}), 400

        input_df = pd.DataFrame([{
            'State': data['State'],
            'Season': data['Season'],
            'Annual_Rainfall': float(data['Annual_Rainfall']),
            'Pesticide': 50.0,
            'Fertiliser': 75.0
        }])

        encoded = ohe_encoder.transform(input_df[['State', 'Season']])
        if hasattr(encoded, 'toarray'):
            encoded = encoded.toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe_encoder.get_feature_names_out(['State', 'Season'])
        )
        final_features = pd.concat([
            encoded_df,
            input_df[['Annual_Rainfall', 'Pesticide', 'Fertiliser']]
        ], axis=1)

        prediction = model.predict(final_features)
        predicted_crop = label_encoder.inverse_transform([prediction[0]])[0]

        return jsonify({'predicted_crop': predicted_crop})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve React frontend
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def fallback(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

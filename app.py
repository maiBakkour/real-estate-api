import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

print("⏳ جاري تحميل الموديل...")

model = joblib.load('price_model.pkl')
with open('model_maps.json', encoding='utf-8') as f:
    maps = json.load(f)

street_map    = maps['street_map']
direction_map = maps['direction_map']
mae           = maps['mae']

condition_map = {'ضعيف': 0, 'وسط': 1, 'جيد': 2, 'ممتاز': 3}

print("✅ الموديل جاهز!")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        street_enc    = street_map.get(data['street'], np.mean(list(street_map.values())))
        direction_enc = direction_map.get(data['direction'], np.mean(list(direction_map.values())))
        condition_num = condition_map.get(data['condition'], 1)

        input_df = pd.DataFrame([{
            'المساحة (م2)'  : float(data['area']),
            'عدد الغرف'     : int(data['rooms']),
            'عدد الحمامات'  : int(data['bathrooms']),
            'floor_num'     : int(data['floor']),
            'condition_num' : condition_num,
            'parking_num'   : 1 if data['has_parking']  else 0,
            'elevator_num'  : 1 if data['has_elevator'] else 0,
            'front_num'     : 1 if data['is_front']     else 0,
            'age'           : int(data.get('age', 20)),
            'apts_floor'    : int(data.get('apts_floor', 4)),
            'street_enc'    : street_enc,
            'direction_enc' : direction_enc,
        }])

        predicted = model.predict(input_df)[0]

        return jsonify({
            'predicted_price': round(float(predicted), 0),
            'min_price'      : round(float(predicted - mae), 0),
            'max_price'      : round(float(predicted + mae), 0),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

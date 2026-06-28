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

street_price      = maps['street_price']
direction_price   = maps['direction_price']
street_ppsqm      = maps['street_ppsqm']
direction_num_map = maps['direction_num_map']
mae               = maps['mae']
mean_price        = maps['mean_price']

condition_map = {'ضعيف': 0, 'وسط': 1, 'جيد': 2, 'ممتاز': 3}

print("✅ الموديل جاهز!")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        area         = float(data['area'])
        rooms        = int(data['rooms'])
        bathrooms    = int(data['bathrooms'])
        floor        = int(data['floor'])
        condition    = data['condition']
        street       = data['street']
        direction    = data['direction'].strip()
        is_front     = bool(data['is_front'])
        has_parking  = bool(data['has_parking'])
        has_elevator = bool(data['has_elevator'])
        age          = int(data.get('age', 20))
        apts_floor   = int(data.get('apts_floor', 4))

        street_enc    = street_price.get(street, mean_price)
        direction_enc = direction_price.get(direction, mean_price)
        ppsqm         = street_ppsqm.get(street, mean_price / 120)
        expected      = area * ppsqm
        direction_num = direction_num_map.get(direction, 0)
        condition_num = condition_map.get(condition, 1)

        input_df = pd.DataFrame([{
            'expected_price': expected,
            'المساحة (م2)'  : area,
            'street_ppsqm'  : ppsqm,
            'street_enc'    : street_enc,
            'عدد الغرف'     : rooms,
            'عدد الحمامات'  : bathrooms,
            'condition_num' : condition_num,
            'front_num'     : 1 if is_front else 0,
            'direction_num' : direction_num,
            'direction_enc' : direction_enc,
            'floor_num'     : floor,
            'age'           : age,
            'apts_floor'    : apts_floor,
            'elevator_num'  : 1 if has_elevator else 0,
            'parking_num'   : 1 if has_parking else 0,
        }])

        predicted = float(model.predict(input_df)[0])

        return jsonify({
            'predicted_price': round(predicted, 0),
            'min_price'      : round(predicted - mae, 0),
            'max_price'      : round(predicted + mae, 0),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'ok',
        'r2'      : maps['r2'],
        'mae'     : mae,
        'version' : '3.0'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

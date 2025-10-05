import os
import joblib

import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from ModelPipeline import create_pipeline
import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = 'Trained Models/XGBoost Model.joblib'


# A simple route
@app.route('/')
def home():
    return "Ahoy! API is alive ⚡"


# Route to upload a file
@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    data = pd.read_csv(file_path).drop('rowid', axis=1, errors='ignore')
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(data)
    classes = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
    prediction = [classes[i] for i in prediction]
    return jsonify({'success': True, 'predictions': prediction})


@app.route('/tweak', methods=['POST'])
def hyper_parameter_tweaking():
    try:
        n_estimators = int(request.args.get('n_estimators', 100))
        learning_rate = float(request.args.get('learning_rate', 0.1))
        max_depth = int(request.args.get('max_depth', 6))
        min_child_weight = float(request.args.get('min_child_weight', 1))
        subsample = float(request.args.get('subsample', 1.0))
        colsample_bytree = float(request.args.get('colsample_bytree', 1.0))
    except KeyError:
        return jsonify({'success': False, 'message': 'One of the parameters is missing'})

    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            new_data = pd.read_csv(file_path).drop('rowid', axis=1, errors='ignore')
            shared_data = pd.read_csv('Shared Data.csv').drop('rowid', axis=1, errors='ignore')
            if not new_data.columns.equals(shared_data.columns):
                return jsonify({'success': False, 'message': 'New Data Cols does not Match Requested Format'})

            if not new_data.dtypes.equals(shared_data.dtypes):
                return jsonify({'success': False, 'message': 'New Data Types does not Match Requested Format'})

            shared_data = pd.concat([shared_data, new_data], ignore_index=True)
            shared_data.drop_duplicates(inplace=True)
            shared_data.to_csv('Shared Data.csv', index=False)

    label_encoder = LabelEncoder()
    data = pd.read_csv('Shared Data.csv').drop('rowid', axis=1, errors='ignore')
    data['koi_disposition_encoded'] = label_encoder.fit_transform(data['koi_disposition'])

    X = data.drop(['koi_disposition', 'koi_disposition_encoded'], axis=1)
    y = data['koi_disposition_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    new_model = XGBClassifier(n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              max_depth=max_depth,
                              min_child_weight=min_child_weight,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree, )
    new_model_pipeline = create_pipeline(new_model)
    new_model_pipeline.fit(X_train, y_train)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"User Trained Models/XGB_Pipeline_{timestamp}.pkl"

    joblib.dump(new_model_pipeline, filename)

    prediction = new_model_pipeline.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    prediction_labels = label_encoder.inverse_transform(prediction)

    corr_matrix = data.corr(numeric_only=True)
    corr_json = {
        'columns': corr_matrix.columns.tolist(),
        'values': corr_matrix.values.tolist()
    }

    conf_matrix = confusion_matrix(y_test_labels, prediction_labels)
    conf_json = {
        'values': conf_matrix.tolist()[::-1],
    }
    return jsonify({'success': True,
                    'accuracy': accuracy_score(y_test, prediction),
                    # 'correlationMatrixInfo': corr_json,
                    'confusionMatrixInfo': conf_json,
                    })


@app.route('/get_header_info', methods=['GET'])
def send_header_info():
    return jsonify({'success': True, 'file_count': len(os.listdir(UPLOAD_FOLDER))})


if __name__ == '__main__':
    app.run(debug=True)
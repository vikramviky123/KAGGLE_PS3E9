from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from src.concrete_strength.a_constants import *
from src.concrete_strength.f_utils.common import load_pickle, read_yaml
from pathlib import Path


app = Flask(__name__)

candidates = {}


@app.route('/')
def index():
    return render_template('Intro.html')


@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/eda/univariate')
def univariate():
    return render_template('univariate.html')


@app.route('/eda/bivariate')
def bivariate():
    return render_template('bivariate.html')


@app.route('/model/modelanalysis')
def modelanalysis():
    return render_template('modelanalysis.html')


def convert_to_float(value):
    return float(value) if value is not None and value != '' else None


@app.route('/model/modelplots')
def modelplots():
    return render_template('modelplots.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:

        # Retrieve the input data from the form
        CementComponent = request.form.get('CementComponent')
        BlastFurnaceSlag = request.form.get('BlastFurnaceSlag')
        FlyAshComponent = request.form.get('FlyAshComponent')
        WaterComponent = request.form.get('WaterComponent')
        SuperplasticizerComponent = request.form.get(
            'SuperplasticizerComponent')
        CoarseAggregateComponent = request.form.get('CoarseAggregateComponent')
        FineAggregateComponent = request.form.get('FineAggregateComponent')
        AgeInDays = request.form.get('AgeInDays')

        # Load the saved list of models using pickle
        best_model_name = 'xgb_regressor'
        pickle_file_path = Path(
            "artifacts/model_trainer/trained_models.joblib")
        loaded_models = load_pickle(pickle_file_path)

        rf_models = loaded_models[best_model_name]

        # Convert form data to float
        def convert_to_float(value):
            return float(value) if value is not None and value != '' else None

        CementComponent = convert_to_float(CementComponent)
        BlastFurnaceSlag = convert_to_float(BlastFurnaceSlag)
        FlyAshComponent = convert_to_float(FlyAshComponent)
        WaterComponent = convert_to_float(WaterComponent)
        SuperplasticizerComponent = convert_to_float(SuperplasticizerComponent)
        CoarseAggregateComponent = convert_to_float(CoarseAggregateComponent)
        FineAggregateComponent = convert_to_float(FineAggregateComponent)
        AgeInDays = convert_to_float(AgeInDays)

        # Create a dictionary from the form data
        data = {
            'CementComponent': [CementComponent],
            'BlastFurnaceSlag': [BlastFurnaceSlag],
            'FlyAshComponent': [FlyAshComponent],
            'WaterComponent': [WaterComponent],
            'SuperplasticizerComponent': [SuperplasticizerComponent],
            'CoarseAggregateComponent': [CoarseAggregateComponent],
            'FineAggregateComponent': [FineAggregateComponent],
            'AgeInDays': [AgeInDays],
        }

        print(data)
        df = pd.DataFrame(data)
        print(df)

        preds = [model.predict(np.array(df)) for model in rf_models]
        print(preds)
        preds_mean = sum(preds) / len(preds)
        print(preds_mean)

        return render_template('predict.html', predicted_concrete_strength=preds_mean[0])

    except Exception as e:
        return render_template('predict.html', error_message=str(e))


@app.route('/form')
def show_form():
    return render_template('predict.html', preds_final=None, error_message=None)


if __name__ == '__main__':
    app.run(debug=True)

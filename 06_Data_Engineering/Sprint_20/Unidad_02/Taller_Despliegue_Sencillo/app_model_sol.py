from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, not enough data to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'predictions': prediction[0]})

# Enruta la funcion al endpoint /api/v1/retrain
@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)

import pickle
import numpy as np
from flask import Flask, render_template, request

app=Flask(__name__)
model=pickle.load(open('model/svc.pkl','rb'))

FEATURES = [
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "smoothness_se",
    "compactness_se",
    "concave_points_se",
    "symmetry_se",
    "texture_worst",
    "smoothness_worst",
    "concave_points_worst",
    "symmetry_worst"
]


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        input_features=[[float(request.form.get(feature,0.0)) for feature in FEATURES]]
        prediction=model.predict(input_features)
        pred_label = "Malignant" if int(prediction) == 1 else "Benign"
        return render_template('index.html',prediction=pred_label)
    except Exception as e:
        return render_template('index.html',prediction=f'Exception occured : {e}')



if __name__ == "__main__":
    app.run(debug = True)
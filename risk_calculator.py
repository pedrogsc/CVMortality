import numpy as np
import tensorflow as tf
from create_models import load_models
from tensorflow.keras.models import load_model
from config import obp_minus, obp_slash, abp_minus, abp_slash


# model_obp = load_model('models/.office10_keras1357.h5')
# model_abp = load_model('models/.ambulatory10_keras1357.h5')

model_obp, model_abp = load_models()


def preprocess_input(input_values, minus, slash):
    return np.array([(float(input_values[key]) - minus[key]) / slash[key] for key in minus])


def predict(model, inputs):
    inputs = np.expand_dims(inputs, axis=0)  # Expand dims to fit model input
    prediction = model.predict(inputs, verbose=0)
    return np.mean(prediction)


def predict_obp(input_values):
    inputs = preprocess_input(input_values, obp_minus, obp_slash)
    return predict(model_obp, inputs)

def predict_abp(input_values):
    inputs = preprocess_input(input_values, abp_minus, abp_slash)
    return predict(model_abp, inputs)

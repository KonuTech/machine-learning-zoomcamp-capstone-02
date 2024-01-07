#!/usr/bin/env python
# coding: utf-8

import logging
from io import BytesIO
from urllib import request as req

import numpy as np
from flask import Flask, request
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image


MODEL_PATH = "models/model_xception_2024-01-06_22-44-37.keras"
model = keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_image(url):
    with req.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_array / 255.0

    return img_preprocessed


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data["url"]
        logging.debug(f"Received URL for prediction: {url}")

        img = download_image(url)
        img = prepare_image(img, target_size=(224, 224))

        if model:
            logging.debug(f"Model loaded successfully from: {MODEL_PATH}")

            score = model.predict(img)
            result = {"prediction": score.tolist()}

            logging.debug(f"Prediction result: {result}")
            if result["prediction"][0][0] > 0.5:
                prediction = "It is a pizza!"
                logging.info(prediction)
                print(prediction)  # Print to terminal
                return prediction
            else:
                prediction = "Not a pizza."
                logging.info(prediction)
                print(prediction)  # Print to terminal
                return prediction
        else:
            logging.error(f"Model file not found at: {MODEL_PATH}")
            return (
                "Model file not found.",
                500,
            )  # HTTP status code 500 for internal server error

    except KeyError as ke:
        logging.exception(f"Key error occurred: {ke}")
        return "Key error occurred.", 400  # Bad request
    except FileNotFoundError as fnfe:
        logging.exception(f"File not found error occurred: {fnfe}")
        return "File not found error occurred.", 404  # Not found
    except Exception as ex:
        logging.exception(f"An exception occurred: {ex}")
        return "An error occurred.", 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

#!/usr/bin/env python
# coding: utf-8

import os
import logging
import grpc
import tensorflow as tf
from flask import Flask, jsonify, request
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from proto import np_to_protobuf

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Adding StreamHandler for printing logs to console
    ]
)

host = os.getenv("TF_SERVING_HOST", "localhost:8500")

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

preprocessor = create_preprocessor("xception", target_size=(224, 224))

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = "pizza-model"
    pb_request.model_spec.signature_name = "serving_default"
    pb_request.inputs["xception_input"].CopyFrom(np_to_protobuf(X))
    return pb_request

def prepare_response(pb_response):
    predictions = pb_response.outputs["dense_9"].float_val
    for prediction in predictions:
        if prediction > 0.5:
            return "It's a pizza!"
        else:
            return "Not a pizza."

def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    logging.debug("Sending prediction request to TensorFlow Serving.")
    pb_response = stub.Predict(pb_request, timeout=20.0)
    logging.debug("Received prediction response from TensorFlow Serving.")
    response = prepare_response(pb_response)
    return response

app = Flask("gateway")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    url = data["url"]
    logging.debug(f"Received URL for prediction: {url}")
    result = predict(url)
    logging.debug(f"Prediction result: {result}")
    return jsonify(result)

if __name__ == "__main__":
    url = "https://m.kafeteria.pl/shutterstock-84904912-9cb8cae338,730,0,0,0.jpg"
    response = predict(url)
    print(response)
    # app.run(debug=True, host="0.0.0.0", port=9696)

FROM tensorflow/serving:2.7.0

COPY pizza-model /models/pizza-model/1
ENV MODEL_NAME="pizza-model"
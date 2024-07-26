import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
import tensorflow as tf
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray


bi_lstm_runner = bentoml.tensorflow.get("bi_lstm_model:latest").to_runner()

svc = bentoml.Service("imdb_bi_lstm", runners=[bi_lstm_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data=np.ndarray) -> np.ndarray:
    result= bi_lstm_runner.run(input_data)
    return result
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Sequential # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import bentoml # type: ignore
from tensorflow.keras.layers import TFSMLayer # type: ignore

model1 = load_model('models\BI_LSTM_model.h5' )
bentoml.tensorflow.save_model('bento_model', model1)
print(f"{model1.tag}")
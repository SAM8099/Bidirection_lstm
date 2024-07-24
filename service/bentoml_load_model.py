from pathlib import Path
import tensorflow as tf
import bentoml

def load_model_and_save_it_in_bentoml(model_file:Path) -> None:
    model = tf.keras.models.load_model(model_file)
    bento_model = bentoml.tensorflow.save_model("bento_keras_model", model)
    print(f"Bento_model : {bento_model.tag}")
    
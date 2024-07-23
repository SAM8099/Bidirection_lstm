from pathlib import Path
import tensorflow as tf
import bentoml

def load_model_and_save_it_in_bentoml(model_file:Path) -> None:
    model = tf.keras.models.load_model(model_file)
    bento_model = bentoml.keras.save_model("bento_keras_model", model)
    print(f"Bento_model : {bento_model.tag}")
    
if __name__=="__main__":
    model_file = Path("BI_LSTM_model.h5")
    load_model_and_save_it_in_bentoml(model_file)
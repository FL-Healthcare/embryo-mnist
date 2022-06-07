import tensorflow as tf
from tensorflow.keras import backend as k
import gc


class FlushCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        #k.clear_session()
       

     
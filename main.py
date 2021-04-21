
import tensorflow as tf
import tfConfig as con
import model

worker_index = 0
tf_config = con.config(worker_index)
strategy = tf.distribute.MultiWorkerMirroredStrategy()
dataset = model.get_dataset()

with strategy.scope():
    worker_model = model.build_and_compile_cnn_model()

worker_model.fit(dataset[0][0], dataset[0][1], epochs=3)
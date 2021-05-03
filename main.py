import tensorflow as tf
import tfConfig as con
import model


#tf_config = con.config()
strategy = tf.distribute.MultiWorkerMirroredStrategy()
global_batch = 64 * 4
dataset = model.get_dataset(global_batch)

with strategy.scope():
    worker_model = model.build_and_compile_cnn_model()

worker_model.fit(dataset, epochs=3, steps_per_epoch=70)

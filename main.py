import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time
import tfConfig as con
import model

NAME = "Mnist-4-rpi-{}".format(int(time.time()))
#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=1)
tf_config = con.config()
com_option = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING) 
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=com_option)
global_batch = 64 * 4
dataset = model.get_dataset(global_batch)

with strategy.scope():
    worker_model = model.build_and_compile_cnn_model()

worker_model.fit(dataset[0], epochs=5, steps_per_epoch=70)
worker_model.evaluate(dataset[1][0],dataset[1][1])

import tensorflow as tf
import tfConfig as con
import model


tf_config = con.config()
com_option = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING) 
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=com_option)
global_batch = 64 * 4
dataset = model.get_dataset(global_batch)

with strategy.scope():
    worker_model = model.build_and_compile_cnn_model()

worker_model.fit(dataset, epochs=3, steps_per_epoch=70)

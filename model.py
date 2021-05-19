import tensorflow as tf


def get_dataset(batch_size):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    #dataset = dataset.shuffle(60000)
    #dataset = dataset.repeat(3)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=60000, count=5))
    dataset = dataset.batch(batch_size)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(buffer_size=1)
    return (dataset,(x_test, y_test))


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
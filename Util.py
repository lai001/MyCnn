import tensorflow as tf


def gpu_growth_session() -> tf.Session:
    return tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

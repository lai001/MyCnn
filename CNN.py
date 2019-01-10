import tensorflow as tf

# 通道
CHANNELS = 3

CONV1_DEEP = 32
CONV1_SIZE = 3

CONV2_DEEP = 64
CONV2_SIZE = 3

CONV3_DEEP = 64
CONV3_SIZE = 3

FC_SIZE = 512

IMAGE_SIZE = {"w": 32, "h": 32}

x_data = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE["w"], IMAGE_SIZE["h"], CHANNELS])
y_data = tf.placeholder(tf.float32, shape=[None, None])


def model(labels_num):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(x_data, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer1-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")

        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # type: tf.Tensor

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope("layer5-full_connect1"):
        full_connect1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        full_connect1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        full_connect1 = tf.nn.relu(tf.matmul(reshaped, full_connect1_weights) + full_connect1_biases)
        full_connect1 = tf.nn.dropout(full_connect1, 0.5)

    with tf.variable_scope("layer6-full_connect2"):
        full_connect2_weights = tf.get_variable('weight', [FC_SIZE, labels_num],
                                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        full_connect2_biases = tf.get_variable("bias", [labels_num], initializer=tf.constant_initializer(0.1))

        output = tf.matmul(full_connect1, full_connect2_weights) + full_connect2_biases

    return output

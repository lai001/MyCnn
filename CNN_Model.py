import tensorflow as tf

# 通道
CHANNELS = 3
# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 3
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3
# 第三层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE = 3
# 全连接层的节点个数
FC_SIZE = 512
# 图像尺寸大小
IMAGE_SIZE = {"w": 64, "h": 64}


x_data = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE["w"], IMAGE_SIZE["h"], CHANNELS])
y_data = tf.placeholder(tf.float32, shape=[None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weightVariable(shape):
    # 过滤器权重变量
    init = tf.random_normal(shape, stddev=0.01)
    # init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    # 建立偏置项变量
    init = tf.random_normal(shape)
    # init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)


def conv2d(x, W):
    # 使用边长为3，深度为32的过滤器，过滤器移动的步长为1.且使用全0填充
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    # 选用最大池化层，池化层过滤器边长为2，使用全0填充且移动的步长为2
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    ''' dropout避免过拟合，使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep大小'''
    return tf.nn.dropout(x, keep)


# 定义卷积神经网络的向前传播过程
def cnnLayer(classnum):
    # 第一层卷积层的尺寸为3*3 ，输入的颜色通道为3 ，深度为32
    W1 = weightVariable([CONV1_SIZE, CONV1_SIZE, CHANNELS, CONV1_DEEP])  # 过滤器权重变量
    b1 = biasVariable([CONV1_DEEP])  # 偏置项
    # 卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为32的过滤器
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)

    pool1 = maxPool(conv1)  # 池化层的向前传播过程,输出32*32*32的矩阵

    # 第二层
    W2 = weightVariable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])  # 过滤器
    b2 = biasVariable([CONV2_DEEP])  # 偏置项
    # 卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为64的过滤器
    conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
    pool2 = maxPool(conv2)  # 池化层的向前传播过程，输出16*16*64的矩阵

    # 第三层
    W3 = weightVariable([CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP])  # 过滤器
    b3 = biasVariable([CONV3_DEEP])  # 偏置项
    # 卷积层的向前传播过程,先使用conv2d卷积函数，再使用RELU激活函数线性变换->非线性变换,使用边长为3，深度为64的过滤器
    conv3 = tf.nn.relu(conv2d(pool2, W3) + b3)
    pool3 = maxPool(conv3)  # 池化层的向前传播过程，输出8*8*64的矩阵

    pool3_shape = pool3.get_shape().as_list()  # 得到矩阵的维度
    nodes = pool3_shape[1] * pool3_shape[2] * pool3_shape[3]  # 计算将矩阵拉直成向量之后的长度，长度为矩阵的长宽和深度的乘积

    # 全连接层
    Wfc = weightVariable([nodes, FC_SIZE])
    bfc = biasVariable([FC_SIZE])
    pool3_reshape = tf.reshape(pool3, [-1, nodes])
    # Wfc = weightVariable([nodes, FC_SIZE])#全过滤器
    # bfc = biasVariable([FC_SIZE])#偏置项
    # pool3_reshape = tf.reshape(pool3, [pool3_shape[0], nodes])#将池化层的输出变成一个batch的向量
    fc = tf.nn.relu(tf.matmul(pool3_reshape, Wfc) + bfc)  # 全连接层的向前传播过程
    dropfc = dropout(fc, keep_prob_75)  # 避免过拟合，随机让某些权重不更新

    # 输出层
    Wout = weightVariable([FC_SIZE, classnum])  # 这一层的输入为一组长度为512的向量，输出一组长度为classnum的向量
    bout = weightVariable([classnum])  # 偏置项
    # out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropfc, Wout), bout)  # 输出层的向前传播过程

    return out

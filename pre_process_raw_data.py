from PIL import Image
import os
from Parameter import *
from Util import *


class _file_notfoundException(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


def generate_tfrecords_file():
    writer = tf.python_io.TFRecordWriter("./faceTF.tfrecords")
    for index, face_dir in enumerate(os.listdir(face_root_dir_path)):

        label = int(face_dir)
        print(label)
        for root, _, files in os.walk(os.path.join(face_root_dir_path, face_dir)):
            os.chdir(root)

            for file in files:
                img = Image.open(file)
                img = img.resize((64, 64))
                img_raw = img.tobytes()  # 将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_tfrecords_file(filename: str, num_epochs: int) -> (tf.Tensor, tf.Tensor):
    # 根据文件名生成一个队列
    queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)  # type: tf.FIFOQueue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)  # type: (tf.Tensor,tf.Tensor)
    # 返回文件名和文件

    if serialized_example.shape == None:
        raise _file_notfoundException(".tfrecords文件没有找到或者不存在")
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                      batch_size=batch_size_value, capacity=5000, num_threads=num_threads,
                                                      min_after_dequeue=2000)
    return image_batch, label_batch


def generate_coordinator_data_labels(file_path: str, num_epochs: int) -> (
        tf.Session, tf.train.Coordinator, tf.Tensor, tf.Tensor):
    image_batch, label_batch = read_tfrecords_file(file_path, num_epochs)  # type: (tf.Tensor, tf.Tensor)
    sess = gpu_growth_session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    return sess, coord, image_batch, label_batch


if __name__ == "__main__":
    if not os.path.exists("faceTF.tfrecords"):
        generate_tfrecords_file()
    else:
        generate_coordinator_data_labels("faceTF.tfrecords", num_epochs=num_epochs)

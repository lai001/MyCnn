from CNN_Model import *
import pre_process_raw_data
import os
from Parameter import *
from Util import *
import sys
import numpy as np


def start_train():
    try:
        while not coord.should_stop():
            img_batch_value, labels_value = sess.run([img_batch, labels])  # type: (np.ndarray,np.ndarray)
            print(img_batch_value.shape,labels_value.shape)
            print(labels_value)
            _, loss_value, steps = sess.run([train, loss, global_steps],
                                            feed_dict={x_data: img_batch_value, y_data: labels_value,
                                                       keep_prob_5: keep_prob_5_value,
                                                       keep_prob_75: keep_prob_75_value})
            if steps % save_interval_step == 0:
                print(f"保存模型到 -> {model_ckpt_path}, 当前保存间隔(每{save_interval_step}步)")
                saver.save(sess, f'{model_ckpt_path}')
                print(f"一共训练了{steps}步")
    except tf.errors.OutOfRangeError:
        print("读取到文件队列末尾,尝试终止所有线程.")
    finally:
        coord.request_stop()
        print('所有线程停止.')


if __name__ == '__main__':

    face_type_num = len(os.listdir(face_root_dir_path))
    global_steps = tf.Variable(0, trainable=False, name="global_steps")
    output = cnnLayer(face_type_num)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_data))
    train = tf.train.AdamOptimizer(base_learning_rate).minimize(loss, global_step=global_steps)
    saver = tf.train.Saver(max_to_keep=saver_max_to_keep)

    sess, coord, img_batch, labels = pre_process_raw_data.generate_coordinator_data_labels(
        "faceTF.tfrecords", num_epochs=num_epochs)

    if not tf.train.checkpoint_exists("model"):
        start_train()

    else:
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, model_ckpt_path)
        print(f"恢复模型，原来已经训练了{sess.run(global_steps)}步")

        op = input("""选择训练或者使用模型进行预测(1：训练   2：预测)""")

        if op is '1':
            start_train()

        elif op is '2':

            image_raw = tf.gfile.FastGFile('Face/15052029/1.png', 'rb').read()

            img = tf.image.decode_jpeg(image_raw)  # type: tf.Tensor
            img1 = tf.image.resize_images(sess.run(img), [64, 64], method=0)
            x = np.array([sess.run(img1)], dtype=np.float32)
            probability = sess.run(output, feed_dict={x_data: x,
                                                      keep_prob_5: keep_prob_5_value,
                                                      keep_prob_75: keep_prob_75_value})  # type: np.ndarray
            print(probability[0][0])
            print(f"概率为{probability}")
        else:
            sys.exit()

    sess.close()
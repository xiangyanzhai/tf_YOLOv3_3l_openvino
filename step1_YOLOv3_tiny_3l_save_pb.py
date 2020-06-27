# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
import tool.yolo_util as yolo_util
from tool.config import Config

base_anchors = [[4, 7], [7, 15], [13, 25], [25, 42], [41, 67], [75, 94], [91, 162], [158, 205], [250, 332]]
base_anchors = tf.constant(base_anchors, dtype=tf.float32)
base_anchors = tf.reshape(base_anchors, (-1, 3, 2))
base_anchors = base_anchors[::-1]


def get_coord(N, stride):
    print(N, stride)
    t = tf.range(int(N / stride))
    x, y = tf.meshgrid(t, t)

    x = x[..., None]
    y = y[..., None]
    coord = tf.concat((x, y, x, y), axis=-1)
    coord = coord[:, :, None, :]
    coord = tf.to_float(coord) * stride
    return coord


def decode_net(net, anchors, coord, stride):
    xy = tf.nn.sigmoid(net[..., :2]) * stride
    wh = tf.exp(net[..., 2:4]) * anchors
    xy1 = xy - wh / 2
    xy2 = xy + wh / 2
    bboxes = tf.concat((xy1, xy2), axis=-1) + coord
    net = tf.nn.sigmoid(net[..., 4:])
    return tf.concat([bboxes, net], axis=-1)


class YOLOv3():
    def __init__(self, config):
        self.config = config
        self.stride = config.stride
        self.blocks = yolo_util.parse_cfg(config.cfg_file)
        self.coords = []
        for i in range(len(self.stride)):
            self.coords.append(get_coord(max(config.img_size), self.stride[i]))

    def build_net_tiny_pb(self):
        self.out = []
        size = self.config.img_size

        im = tf.placeholder(dtype=tf.float32, shape=(1, self.config.img_size[0], self.config.img_size[1], 3),
                            name='img')
        self.n_im = im
        m = tf.shape(im)[0]

        with slim.arg_scope([slim.batch_norm], is_training=False, scale=True, decay=0.9, epsilon=1e-5):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                Name, end_points = yolo_util.create_net(im, self.blocks)
        self.Name = Name

        for i in range(len(Name)):
            print(i, '\t', Name[i], '\t', end_points[i])

        net13 = end_points[16]
        net26 = end_points[23]
        net52 = end_points[-1]
        map_H = tf.shape(net13)[1]
        map_W = tf.shape(net13)[2]
        self.map_H = map_H
        # map_H = int(self.config.img_size / 32)
        # map_W = int(416 / 32)
        C = [net13, net26, net52]
        decode = []
        for i in range(3):
            # tH = map_H * 2 ** i
            # tW = map_W * 2 ** i
            tH, tW = tf.shape(C[i])[1], tf.shape(C[i])[2]
            decode.append(tf.reshape(decode_net(tf.reshape(C[i], (-1, tH, tW, 3, self.config.num_cls + 5)),
                                                base_anchors[i], self.coords[i][:tH, :tW], self.stride[i]),
                                     (m, -1, self.config.num_cls + 5)))
        decode = tf.concat(decode, axis=1)
        print(decode)
        self.out.append(decode.name[:-2])
        print(self.out)
        print(size, self.config.num_cls)

        # self.result = predict(decode[0], size, self.config.num_cls, c_thresh=0.005, iou_thresh_=0.45)

    def init(self, sess, Name, init_vars, W):
        # W = yolo_utils.load_weights(self.config.weight_file, self.blocks, Name, init_vars, (80 + 5) * 5, count=5)
        print('***********', len(W), len(init_vars))
        for i in range(len(W)):
            weight = W[i]
            if init_vars[i].get_shape().as_list()[-1] != (self.config.num_cls + 5) * 3:
                print(i, init_vars[i].name, init_vars[i].shape, weight.shape)
                sess.run(init_vars[i].assign(weight))
            else:
                sess.run(init_vars[i].assign(weight))
                print('*************************')
        print('init finish')

    def save_pb(self):
        self.build_net_tiny_pb()
        # file = self.config.pre_model
        # file='/home/zhai/PycharmProjects/Demo35/myDNN/SSD_tf/train/models/SSD300_4.ckpt-60000'

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        W = yolo_util.load_weights(self.config.weight_file, self.blocks, self.Name, tf.global_variables(),
                                   (self.config.num_cls + 5) * 3, count=5)

        print(len(W), len(tf.global_variables()))
        with tf.Session(config=config) as sess:
            self.init(sess, self.Name, tf.global_variables(), W)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, self.out)
            with tf.gfile.FastGFile('./models/YOLOv3_tiny_3l_20_run_model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        print('finish')


if __name__ == "__main__":
    files = None
    cfg_file = r'D:\PycharmProjects\daknet_demo\train\brand_20_classes_train\cfg\yolov3-tiny_3l_brand_20.cfg'
    weights_file = r'D:\PycharmProjects\daknet_demo\train\brand_20_classes_train\backup\yolov3-tiny_3l_brand_20_25000.weights'

    config = Config(files, cfg_file, weights_file, num_cls=20, img_size=(416, 416), ignore_thresh=0.5,
                    stride=[32, 16, 8],
                    equal_scale=False)

    yolov3 = YOLOv3(config)
    # yolov3.test()
    # yolov3.test_coco()
    # yolov3.test_tiny()
    # yolov3.test_vedio()
    yolov3.save_pb()

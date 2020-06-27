# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
import cv2 as cv
from openvino.inference_engine import IENetwork, IEPlugin

class BrandRecognition():

    def __init__(self, yolo_model_xml, yolo_model_bin, cpu_extension_file, ):
        self.num_cls = 20
        self.thr = 0.5
        yolo_net = IENetwork(model=yolo_model_xml, weights=yolo_model_bin)
        self.yolo_input_blob = next(iter(yolo_net.inputs))
        self.yolo_out_blob = next(iter(yolo_net.outputs))

        plugin = IEPlugin(device='CPU')
        plugin.add_cpu_extension(cpu_extension_file)

        # else:
        #     plugin = IEPlugin(device='GPU')
        # plugin = IEPlugin(device='GPU',
        #                   plugin_dirs=r'C:\Program Files (x86)\IntelSWTools\openvino_2019.3.334\deployment_tools\inference_engine\bin\intel64\Release/')

        exec_yolo_net = plugin.load(network=yolo_net)

        del yolo_net

        self.exec_yolo_net = exec_yolo_net

    def detect_img(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)

        yolo_res = self.exec_yolo_net.infer(inputs={self.yolo_input_blob: blob})
        outs = yolo_res[self.yolo_out_blob][0]
        conf = outs[:, 4]


        thresh = 0.01
        thresh = max(np.sort(conf)[::-1][100], thresh)
        inds = conf > thresh
        outs = outs[inds]
        outs[:, 5:] = outs[:, 4:5] * outs[:, 5:]

        outs[:, :4] = np.clip(outs[:, :4], 0, 416)

        outs[:, :4] = outs[:, :4] / 416 * np.array([w, h, w, h])
        outs[:, 2:4] = outs[:, 2:4] - outs[:, :2]

        bboxes = outs[:, :4].tolist()
        for i in range(self.num_cls):
            score = outs[:, 5 + i].copy()
            inds = cv.dnn.NMSBoxes(bboxes, score, 0.005, 0.45)
            inds = np.array(inds).ravel().astype(np.int32)
            score[inds] *= -1
            inds = score > 0
            score[inds] = 0
            score *= -1
            outs[:, 5 + i] = score
        cls = np.argmax(outs[:, 5:], axis=-1)

        score = outs[:, 5:].max(axis=-1)
        outs[:, 4] = score
        outs[:, 5] = cls
        inds = score > self.thr
        outs = outs[inds]
        outs[:, 2:4] += outs[:, :2]
        inds = outs[:, 4].argsort()[::-1]
        outs = outs[inds]
        outs = outs[:, :6]
        return outs

    def filter_bboxes(self, bboxes, conf=0.5, min_h=22, min_w=27):
        # bboxes  xyxy score cls
        inds = bboxes[:, 4] > conf
        bboxes = bboxes[inds]
        wh = bboxes[:, 2:4] - bboxes[:, :2]
        inds = (wh[:, 0] >= min_w) & (wh[:, 1] >= min_h)
        bboxes = bboxes[inds]
        return bboxes

    def detect_recognition(self, img):
        # BGR
        res = self.detect_img(img)
        # res = self.filter_bboxes(res)
        # bboxes = np.round(res[:, :4])
        # return bboxes
        return res

    # def draw_word(im, bboxes, R, sleep=1):


#     # im = im[..., ::-1]
#     im = im.astype(np.uint8)
#     boxes = bboxes.astype(np.int32)
#     i = 0
#     for box in boxes:
#         if R[i] == '#':
#             continue
#         x1, y1, x2, y2 = box[:4]
#
#         im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
#         im = cv2.putText(im, R[i], (x1, y1), font, 1., (0, 255, 255), 2)
#         i += 1
#     im = im.astype(np.uint8)
#     cv2.imshow('img', im)
#     cv2.waitKey(sleep)
#     return im


if __name__ == "__main__":
    yolo_model_xml = r'./models\YOLOv3_tiny_3l_20_run_model.xml'
    yolo_model_bin = r'./models\YOLOv3_tiny_3l_20_run_model.bin'

    cpu_extension_file = r'C:\Users\gpu3\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\cpu_extension.dll'

    brandrecognition = BrandRecognition(yolo_model_xml, yolo_model_bin, cpu_extension_file)
    font = cv2.FONT_HERSHEY_SIMPLEX
    import os
    import time
    from datetime import datetime

    cap = cv2.VideoCapture(r'D:\datasets/VID_20200426_163347.mp4')
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        bboxes = brandrecognition.detect_img(frame)

        for bbox in bboxes:
            x1, y1, x2, y2, score, cls = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255))
            frame = cv2.putText(frame, str(int(cls)), (x1, y1), font, 1., (0, 0, 0), 2)
        cv2.imshow('img', frame)
        cv2.waitKey(1)

# python mo_tf.py   --input_model D:\PycharmProjects\Demo36\CRNN\test_9\CRNN_model.pb --data_type FP32
#
# python mo_tf.py   --input_model D:\PycharmProjects\Demo36\YOLOv3_tiny\test_2\YOLOv3_tiny_run_model.pb --data_type FP32


# python mo_tf.py   --input_model D:\PycharmProjects\Demo36\CRNN\test_9\CRNN_model.pb --data_type FP16  --output_dir D:\PycharmProjects\Demo36\BrandRecognition\models_16
#
# python mo_tf.py   --input_model D:\PycharmProjects\Demo36\YOLOv3_tiny\test_2\YOLOv3_tiny_run_model.pb --data_type FP16 --output_dir D:\PycharmProjects\Demo36\BrandRecognition\models_16

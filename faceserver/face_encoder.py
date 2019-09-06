# -*- coding: utf-8 -*-
# @file face_encoder.py
# @author lidongming1@360.cn
# @date 2019-08-23 15:59
# @brief

import pickle
import os
import cv2
import base64

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet.src.facenet as face_net
import facenet.src.align.detect_face

import time

np.set_printoptions(suppress=True)
gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.abspath("./models/facenet/20180408-102900")

class Encoder:  
    def __init__(self):
        self.dectection= Detection()
        self.sess = tf.Session()
        start=time.time()
        with self.sess.as_default():
            face_net.load_model(facenet_model_checkpoint)
        print 'Model loading finised,cost: %ds'%((time.time()-start))

    def generate_embedding(self, image):
        res = []
        # enables multi-threading
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        face=self.dectection.find_faces(image)
        if not face:
            return res

        prewhiten_face = face_net.prewhiten(face.image)
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]

    def generate_embedding_from_base64(self, image_base64):
        res = []
        # enables multi-threading
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        face=self.dectection.find_faces_from_base64(image_base64)

        if not face:
            return res

        prewhiten_face = face_net.prewhiten(face.image)
        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        res = self.sess.run(embeddings, feed_dict=feed_dict)[0]

        #  print 'start join res'
        #  res_str = ','.join(map(str, res))
        #  return res_str
        return res

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return facenet.src.align.detect_face.create_mtcnn(sess, None)

    def get_image(self, image):
        return misc.imread(os.path.expanduser(image), mode='RGB')

    def get_image_from_base64(self, image_base64):
        image = []
        try:
            img_data = base64.b64decode(image_base64)
            nparray = np.fromstring(img_data, np.uint8)
        except:
            return image

        image = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_faces(self, image):
        faces = []

        bounding_boxes, _ = facenet.src.align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)
        return faces

    def find_faces(self, image_file):
        #  image = misc.imread(os.path.expanduser(image_file), mode='RGB')
        image = self.get_image(image_file)
        faces = self.get_faces(image)
        return faces[0]

    def find_faces_from_base64(self, image_base64):
        image = self.get_image_from_base64(image_base64)
        if len(image):
            faces = self.get_faces(image)
            if len(faces):
                return faces[0]
            else:
                return None
        else:
            return None

if __name__=='__main__':
    #pic='test.jpg'
    #encoder = Encoder()
    ##  print encoder.generate_embedding(pic)

    ## with open(pic, "rb") as image_file:
    ##     image_base64 = base64.b64encode(image_file.read())
    ##     #  image_base64 = image_base64.decode('utf-8')
    ##     print encoder.generate_embedding_from_base64(image_base64)

    #with open("b.txt") as f:
    #    line = f.readline()
    #    print encoder.generate_embedding_from_base64(line)

    sample_path='./samples'
    encoder = Encoder()
    samples=os.listdir(sample_path)
    for image_file in samples:
        path = os.path.join(sample_path, image_file)
        print path
        print encoder.generate_embedding(path)

import os
import cv2
import pickle
from tqdm import trange
import numpy as np

from odometry.utils import mlflow_logging


class BoVW:

    @mlflow_logging(name='BoW', prefix='model.')
    def __init__(self, clusters_num=64, knn=20, feature='SIFT'):

        if feature == 'SIFT':
            self.extractor = cv2.xfeatures2d.SIFT_create()
        else:
            raise RuntimeError('No other type of features except SIFT is implemented')

        self.knn_matcher = cv2.BFMatcher()
        self.clusters_num = clusters_num
        self.BoVW = cv2.BOWKMeansTrainer(clusters_num)
        self.voc = None
        self.descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.extractor, self.knn_matcher)
        self.knn = knn

        self.histograms = list()
        self.images = list()
        self.matches = list()
        self.counter = 0

    def fit(self, generator):

        for _ in trange(len(generator)):
            x, _ = next(generator)
            images = x[0]
            for i in range(images.shape[0]):
                image = np.uint8(images[i])
                kp, des = self.extractor.detectAndCompute(image, None)
                self.BoVW.add(des) if des else None

        self.voc = self.BoVW.cluster()
        self.descriptor_extractor.setVocabulary(self.voc)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.voc, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.voc = pickle.load(f)
            self.descriptor_extractor.setVocabulary(self.voc)

    @staticmethod
    def ratio_test(matches):
        'David G. Lowe. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vision, 60(2):91–110'
        'November 2004.'
        ratio_threshold = 0.7
        good_matches = list()
        for match in matches:
            if match[0].distance < ratio_threshold * match[1].distance:
                good_matches.append(match)
        return good_matches

    def keypoints_overlap_test(self, match, des1):

        matches_threshold = 10

        good_matches = list()
        for k in range(len(match[0])):
            ind = match[0][k].trainIdx

            image = np.uint8(self.images[ind])

            kp2, des2 = self.extractor.detectAndCompute(image, None)

            descriptors_match = self.knn_matcher.knnMatch(des2, des1, 2)
            good_descriptors_match = self.ratio_test(descriptors_match)

            if len(good_descriptors_match) > matches_threshold:
                good_matches.append((match[0][k], len(good_descriptors_match)))

        good_matches.sort(key=lambda tup: tup[1], reverse=True)

        return good_matches

    def predict(self, image: np.ndarray, robust=True):

        self.images.append(image)
        image = np.uint8(image)
        kp, des = self.extractor.detectAndCompute(image, None)
        hist = self.descriptor_extractor.compute(image=image, keypoints=kp)

        if self.counter > 0:
            match = self.knn_matcher.knnMatch(hist, np.vstack(self.histograms), min(self.counter, self.knn))
            match = self.keypoints_overlap_test(match, des) if robust else None
        else:
            match = list()

        self.histograms.append(hist)
        self.matches.append(match)
        self.counter += 1

        return match

    def clear(self):
        self.histograms = list()
        self.images = list()
        self.matches = list()
        self.counter = 0

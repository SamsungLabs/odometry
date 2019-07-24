import os
import cv2
import pickle
from tqdm import trange
import numpy as np


class BoVW():
    def __init__(self, clusters_num=64):

        self.extractor = cv2.xfeatures2d.SIFT_create()
        self.knn_matcher = cv2.BFMatcher()
        self.clusters_num = clusters_num
        self.BoVW = cv2.BOWKMeansTrainer(clusters_num)
        self.voc = None
        self.descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.extractor, self.knn_matcher)
        self.knn = 20

        self.histograms = list()
        self.key_frames = list()
        self.matches = list()
        self.counter = 0

    def fit(self, generator):

        for i in trange(len(generator)):
            x, y = next(generator)
            for b in range(x[0].shape[0]):
                    kp, des = self.extractor.detectAndCompute(np.uint8(x[0][b]), None)
                    if des is not None:
                        self.BoVW.add(des)

        self.voc = self.BoVW.cluster()
        self.descriptor_extractor.setVocabulary(self.voc)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.voc, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.voc = pickle.load(f)
            self.descriptor_extractor.setVocabulary(self.voc)

    def ratio_test(self, matches):
        'David G. Lowe. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vision, 60(2):91–110'
        'November 2004.'
        ratio_treshold = 0.7
        good_matches = list()
        for match in matches:
            if match[0].distance < ratio_treshold * match[1].distance:
                good_matches.append(match)
        return good_matches

    def keypoints_overlap_test(self, match, des1):

        matches_treshold = 10

        good_matches = list()
        for k in range(len(match[0])):
            ind = match[0][k].trainIdx

            img = self.key_frames[ind]

            kp2, des2 = self.extractor.detectAndCompute(np.uint8(img), None)

            descriptors_match = self.knn_matcher.knnMatch(des2, des1, 2)
            good_descriptors_match = self.ratio_test(descriptors_match)

            if len(good_descriptors_match) > matches_treshold:
                good_matches.append((match[0][k], len(good_descriptors_match)))

        good_matches.sort(key=lambda tup: tup[1], reverse=True)

        return good_matches

    def predict(self, image: np.ndarray, robust=True):

        self.key_frames.append(image)
        kp, des = self.extractor.detectAndCompute(np.uint8(image), None)
        hist = self.descriptor_extractor.compute(image=np.uint8(image), keypoints=kp)

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
        self.key_frames = list()
        self.matches = list()
        self.counter = 0

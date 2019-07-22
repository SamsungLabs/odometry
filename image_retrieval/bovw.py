import os
import cv2
import pickle
from tqdm import tqdm, trange
import numpy as np



class BoVW():
    def __init__(self, clusters_num=64):

        self.extractor = cv2.xfeatures2d.SIFT_create()
        self.clusters_num = clusters_num
        self.BoVW = cv2.BOWKMeansTrainer(clusters_num)
        self.voc = None
        extractor = cv2.xfeatures2d.SIFT_create()
        flann_params = dict(algorithm=1, trees=5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.descriptor_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
        self.knn_matcher = cv2.FlannBasedMatcher(flann_params)
        self.knn = 50

    def fit(self, generator):

        for i in trange(len(generator)):
            x, y = next(generator)
            for b in range(x[0].shape[0]):
                    kp, des = self.extractor.detectAndCompute(np.uint8(x[0][b]), None)
                    if des is not None:
                        for d in des:
                            self.BoVW.add(d)

        self.voc = self.BoVW.cluster()
        self.descriptor_extractor.setVocabulary(self.voc)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.voc, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.voc = pickle.load(f)
            self.descriptor_extractor.setVocabulary(self.voc)

    def predict(self, generator):

        result = list()
        keypoints = list()
        histograms = list()

        self.knn_matcher.clear()

        counter = 0
        for i in trange(len(generator)):
                x, y = next(generator)
                for b in range(x[0].shape[0]):
                    counter += 1
                    if not(counter % 10):
                        kp = self.extractor.detect(np.uint8(x[0][b]))
                        keypoints.append(kp)
                        hist = self.descriptor_extractor.compute(image=np.uint8(x[0][b]), keypoints=kp)
                        histograms.append(hist)
                        matches = self.knn_matcher.knnMatch(hist, self.knn)
                        result.append(matches)
                        self.knn_matcher.add(hist)
                        self.knn_matcher.train()
        return result, keypoints, histograms

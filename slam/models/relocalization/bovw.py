import cv2
import mlflow
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import trange

from slam.utils import mlflow_logging
from slam.utils import resize_image


class BoVW:

    @mlflow_logging(ignore=('run_dir',), name='BoW', prefix='model.')
    def __init__(self, clusters_num=64,
                 knn=20,
                 matches_threshold=10,
                 min_descriptors_num=10,
                 feature='SIFT',
                 matcher='BruteForce',
                 run_dir=None):

        if feature == 'SIFT':
            self.extractor = cv2.xfeatures2d.SIFT_create()
        else:
            raise RuntimeError('No other type of features except SIFT is implemented')

        if matcher == 'BruteForce':
            self.knn_matcher = cv2.BFMatcher()
        else:
            flann_params = dict(algorithm=1, trees=5)
            self.knn_matcher = cv2.FlannBasedMatcher(flann_params, {})

        self.clusters_num = clusters_num
        self.BoVW = cv2.BOWKMeansTrainer(clusters_num)
        self.voc = None
        self.descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.extractor, self.knn_matcher)
        self.knn = knn
        self.matches_threshold = matches_threshold

        self.histograms = None
        self.images = None
        self.matches = None
        self.counter = None

        self.clear()

        self.index_mapping = dict()

        self.run_dir = run_dir

        self.min_descriptors_num = min_descriptors_num

    def fit(self, generator):

        for _ in trange(len(generator)):
            x, _ = next(generator)
            images = x[0]
            for i in range(images.shape[0]):
                image = np.uint8(images[i])
                kp, des = self.extractor.detectAndCompute(image, None)
                self.BoVW.add(des) if des is not None else None

        self.voc = self.BoVW.cluster()
        self.descriptor_extractor.setVocabulary(self.voc)

        self.save(Path(self.run_dir) / f'vocabulary.pkl') if self.run_dir else None

    def save(self, path):

        if not isinstance(path, Path):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path.as_posix(), 'wb') as f:
            pickle.dump(self.voc, f)

        mlflow.log_artifacts(self.run_dir) if mlflow.active_run() else None

    def load(self, path):

        if not isinstance(path, Path):
            path = Path(path)

        with open(path.as_posix(), 'rb') as f:
            self.voc = pickle.load(f)
            self.descriptor_extractor.setVocabulary(self.voc)

    @staticmethod
    def ratio_test(matches):
        """David G. Lowe. Distinctive image features from scale-invariant keypoints. Int. J. Comput. Vision,
         60(2):91–110, November 2004."""
        ratio_threshold = 0.7
        good_matches = list()
        for match in matches:
            if len(match) < 2:
                continue
            if match[0].distance < ratio_threshold * match[1].distance:
                good_matches.append(match)
        return good_matches

    def keypoints_overlap_test(self, match, des1):

        knn_matches_num = 2
        good_matches = list()
        if match is None or len(match) == 0 or des1 is None or len(des1) < knn_matches_num:
            return list()

        for k in range(len(match[0])):
            index = match[0][k].trainIdx

            image = np.uint8(self.images[index])

            kp2, des2 = self.extractor.detectAndCompute(image, None)

            if des2 is None or len(des2) < knn_matches_num:
                list()

            descriptors_match = self.knn_matcher.knnMatch(des2, des1, knn_matches_num)
            good_descriptors_match = self.ratio_test(descriptors_match)

            if len(good_descriptors_match) > self.matches_threshold:
                good_matches.append((match[0][k], len(good_descriptors_match)))

        good_matches.sort(key=lambda tup: tup[1], reverse=True)

        return good_matches

    def predict(self, image: np.ndarray, index: int, robust: bool = True):

        hist, des = self.add(image, index)

        if hist is None or len(hist) == 0 or self.counter == 1:
            return pd.DataFrame(columns=['to_db_index', 'from_db_index', 'to_index', 'from_index', 'matches_num'])

        histograms = np.vstack(self.histograms[:-1])
        assert histograms.shape[0] >= min(self.counter - 1, self.knn)
        try:
            match = self.knn_matcher.knnMatch(hist, histograms, min(self.counter - 1, self.knn))
            match = self.keypoints_overlap_test(match, des) if robust else match
        except Exception as e:
            np.save('histograms.npy', histograms)
            print('Error in BoVW class. Histograms has been dumped')
            print(e)
            return pd.DataFrame(columns=['to_db_index', 'from_db_index', 'to_index', 'from_index', 'matches_num'])

        df = pd.DataFrame({'to_db_index': [self.counter - 1] * len(match),
                           'from_db_index': [m[0].trainIdx for m in match],
                           'to_index': [index] * len(match),
                           'from_index': [self.index_mapping[m[0].trainIdx] for m in match],
                           'matches_num': [m[1] for m in match]})

        self.matches = self.matches.append(df)

        return df

    def add(self, image, index):
        height, width, channels_num = image.shape
        small_height = height // 4
        small_width = width // 4
        image = resize_image(image, (small_width, small_height))
        image = np.uint8(image)
        kp, des = self.extractor.detectAndCompute(image, None)

        if des is None or len(des) < self.min_descriptors_num:
            return None, None

        hist = self.descriptor_extractor.compute(image=image, keypoints=kp)

        if hist is None or len(hist) == 0:
            return None, None

        self.index_mapping[self.counter] = index
        self.images.append(image)
        self.histograms.append(hist)
        self.counter += 1

        return hist, des

    def clear(self):
        self.histograms = list()
        self.images = list()
        self.matches = pd.DataFrame(columns=['to_db_index', 'from_db_index', 'to_index', 'from_index', 'matches_num'])
        self.counter = 0

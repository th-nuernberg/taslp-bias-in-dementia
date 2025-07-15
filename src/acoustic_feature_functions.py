"""
@author: Kashaf Gulzar

File to compute features used in acoustic_feature_extraction.py
"""
# -------------- Import Modules ----------------
from librosa.feature import mfcc
import opensmile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# -------------- Features Functions ----------------


class GetFeatures:
    """
    Creates a class for computing embeddings and PCA for every audio file
    """
    def __init__(self):
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def get_mfcc(self, data, fs):
        feat = mfcc(y=data, sr=fs, n_mfcc=40)
        mfcc_matrix = feat[1:, :]
        mfcc_1d = list(feat.mean(axis=1))
        mfcc_1d = mfcc_1d[1:]
        return mfcc_matrix, mfcc_1d

    def get_egemaps(self, data, fs):
        feat = self.smile.process_signal(data, fs)
        egemap = list(feat.iloc[0, 0:])
        return egemap

    def get_embedding(self, data, fs):
        feat_mfcc = self.get_mfcc(data,fs)
        feat_egemaps = self.get_egemaps(data,fs)
        embedding = feat_mfcc+feat_egemaps
        return embedding

    def get_pca(self, embedding):
        scaled_embedding = self.scaler.fit_transform(np.array(list(embedding.values())))
        pca_embedding = self.pca.fit_transform(scaled_embedding)
        tagged_embedding = zip(embedding.keys(),pca_embedding)
        return tagged_embedding




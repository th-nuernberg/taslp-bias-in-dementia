"""
@author: Kashaf Gulzar

Class Metadata which contains demographic information,
Class Test data creates objects for every audio file with all the embeddings and metadata for classification
Class Test data Plot creates PCA objects for every audio file with all the embeddings and metadata for plotting use case
Class plot contains functions for generating different plots
"""


class MetaData:
    def __init__(self, f_name, age, sex, mmse, group, depression, depression_score):
        self.file = f_name
        self.age = age
        self.sex = sex
        self.mmse = mmse
        self.group = group
        self.depression = depression
        self.depression_score = depression_score

    def is_female(self):
        return self.sex == 0

    def is_male(self):
        return self.sex == 1


class TestData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.mfcc_matrix = None
        self.mfcc_1d = None
        self.egemaps = None
        self.latentlayer = None
        self.hiddenlayer_1 = None
        self.hiddenlayer_2 = None
        self.hiddenlayer_3 = None
        self.hiddenlayer_4 = None
        self.hiddenlayer_5 = None
        self.hiddenlayer_6 = None
        self.hiddenlayer_7 = None
        self.hiddenlayer_8 = None
        self.hiddenlayer_9 = None
        self.hiddenlayer_10 = None
        self.hiddenlayer_11 = None
        self.hiddenlayer_12 = None
        self.hiddenlayer_13 = None
        self.metadata = None

    def set_mfcc_matrix(self, mfcc_matrix):
        self.mfcc_matrix = mfcc_matrix

    def set_mfcc_1d(self, mfcc_1d):
        self.mfcc_1d = mfcc_1d

    def set_egemaps(self, egemaps):
        self.egemaps = egemaps

    def set_latentlayer(self, latentlayer):
        self.latentlayer = latentlayer

    def set_hiddenlayer_1(self, hiddenlayer_1):
        self.hiddenlayer_1 = hiddenlayer_1

    def set_hiddenlayer_2(self, hiddenlayer_2):
        self.hiddenlayer_2 = hiddenlayer_2

    def set_hiddenlayer_3(self, hiddenlayer_3):
        self.hiddenlayer_3 = hiddenlayer_3

    def set_hiddenlayer_4(self, hiddenlayer_4):
        self.hiddenlayer_4 = hiddenlayer_4

    def set_hiddenlayer_5(self, hiddenlayer_5):
        self.hiddenlayer_5 = hiddenlayer_5

    def set_hiddenlayer_6(self, hiddenlayer_6):
        self.hiddenlayer_6 = hiddenlayer_6

    def set_hiddenlayer_7(self, hiddenlayer_7):
        self.hiddenlayer_7 = hiddenlayer_7

    def set_hiddenlayer_8(self, hiddenlayer_8):
        self.hiddenlayer_8 = hiddenlayer_8

    def set_hiddenlayer_9(self, hiddenlayer_9):
        self.hiddenlayer_9 = hiddenlayer_9

    def set_hiddenlayer_10(self, hiddenlayer_10):
        self.hiddenlayer_10 = hiddenlayer_10

    def set_hiddenlayer_11(self, hiddenlayer_11):
        self.hiddenlayer_11 = hiddenlayer_11

    def set_hiddenlayer_12(self, hiddenlayer_12):
        self.hiddenlayer_12 = hiddenlayer_12

    def set_hiddenlayer_13(self, hiddenlayer_13):
        self.hiddenlayer_13 = hiddenlayer_13

    def set_meta_data(self, meta):
        self.metadata = meta


class TestDataPlot:
    def __init__(self, file_name):
        self.file_name = file_name
        self.pca_mfcc = None
        self.pca_egemaps = None
        # self.pca_combined = None
        self.pca_wav2vec_lastlayer = None
        self.pca_wav2vec_latent = None
        self.metadata = None

    def set_pca_mfcc(self, pca_mfcc):
        self.pca_mfcc = pca_mfcc

    def set_pca_egemaps(self, pca_egemaps):
        self.pca_egemaps = pca_egemaps

    # def set_pca_combined(self, pca_combined):
    #     self.pca_combined = pca_combined

    def set_pca_wav2vec_lastlayer(self, pca_wav2vec_lastlayer):
        self.pca_wav2vec_lastlayer = pca_wav2vec_lastlayer

    def set_pca_wav2vec_latent(self, pca_wav2vec_latent):
        self.pca_wav2vec_latent = pca_wav2vec_latent

    def set_meta_data(self, meta):
        self.metadata = meta

    def is_female(self):
        if self.metadata is None:
            return False
        return self.metadata.is_female()

    def is_male(self):
        if self.metadata is None:
            return False
        return self.metadata.is_male()


class Plot:
    @staticmethod
    def plot_gender(gender,val_pair,plt):
        if gender == 0: # female
            color = 'r'
        elif gender == 1: # Male
            color = 'b'
        plt.scatter(val_pair[0],val_pair[1], c=color)

    @staticmethod
    def plot_age(age,val_pair,plt):
        if 0<= age <=65: # age group 1
            color = 'r'
        elif 66<= age <=100: # age group 2
            color = 'b'
        plt.scatter(val_pair[0],val_pair[1], c=color)

    @staticmethod
    def plot_adgroup(ad_group,val_pair,plt):
        if ad_group == 0: # Healthy
            color = 'b'
        elif ad_group == 1: # AD_patient
            color = 'r'
        plt.scatter(val_pair[0],val_pair[1], c=color)

    @staticmethod
    def plot_ad(ad_severity,val_pair,plt):
        if 0<= ad_severity <=15: # Alzheimers severe
            color = 'r'
        elif 16<= ad_severity <=23: # Alzheimers mild
            color = 'b'
        elif 24<= ad_severity <=30: # Alzheimers low
            color = 'c'
        plt.scatter(val_pair[0],val_pair[1], c=color)

    @staticmethod
    def plot_depression(d_status,val_pair,plt):
        if d_status == 0: # No depression
            color = 'b'
        elif d_status == 1: # depression
            color = 'r'
        plt.scatter(val_pair[0],val_pair[1], c=color)
"""
@author: Kashaf Gulzar

Dataloader file to load pickle file of features and preprocess it.
"""
# -------------- Import Modules ----------------
import pandas as pd
import numpy as np
import pickle
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from itertools import chain

# -------------- Imports and preprocesses data ----------------


class Data:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def import_data(self, embedding_path):
        age, gender, file_name, ad_status, mmse_score, depression_status, latent_embedding, mfcc_1d_embedding, \
        egemaps_embedding, hidden_layer_1, hidden_layer_2, hidden_layer_3, hidden_layer_4, hidden_layer_5, hidden_layer_6,\
        hidden_layer_7, hidden_layer_8, hidden_layer_9, hidden_layer_10, hidden_layer_11, hidden_layer_12, hidden_layer_13 \
        = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        with open(embedding_path, 'rb') as handle:
            data = pickle.load(handle)

        for f_name, t_file in data.items():
            file_name.append(f_name)
            mfcc_1d_embedding.append(t_file.mfcc_1d)
            egemaps_embedding.append(t_file.egemaps)
            latent_embedding.append(t_file.latentlayer)
            hidden_layer_1.append(t_file.hiddenlayer_1)
            hidden_layer_2.append(t_file.hiddenlayer_2)
            hidden_layer_3.append(t_file.hiddenlayer_3)
            hidden_layer_4.append(t_file.hiddenlayer_4)
            hidden_layer_5.append(t_file.hiddenlayer_5)
            hidden_layer_6.append(t_file.hiddenlayer_6)
            hidden_layer_7.append(t_file.hiddenlayer_7)
            hidden_layer_8.append(t_file.hiddenlayer_8)
            hidden_layer_9.append(t_file.hiddenlayer_9)
            hidden_layer_10.append(t_file.hiddenlayer_10)
            hidden_layer_11.append(t_file.hiddenlayer_11)
            hidden_layer_12.append(t_file.hiddenlayer_12)
            hidden_layer_13.append(t_file.hiddenlayer_13)
            mmse_score.append(t_file.metadata.mmse)
            ad_status.append(t_file.metadata.group)
            depression_status.append(t_file.metadata.depression)
            age.append(t_file.metadata.age)
            gender.append(t_file.metadata.sex)

        df = pd.DataFrame(list(zip(file_name, age, gender, mfcc_1d_embedding, egemaps_embedding, latent_embedding,
                                   hidden_layer_1, hidden_layer_2, hidden_layer_3, hidden_layer_4, hidden_layer_5, hidden_layer_6,
                                   hidden_layer_7, hidden_layer_8, hidden_layer_9, hidden_layer_10, hidden_layer_11, hidden_layer_12,
                                   hidden_layer_13, ad_status, depression_status, mmse_score)),
                          columns=['Filename', 'Age', 'Gender', 'MFCC', 'egemaps', 'Wav2vec_latent', 'Wav2vec_Hidden_1',
                                   'Wav2vec_Hidden_2', 'Wav2vec_Hidden_3', 'Wav2vec_Hidden_4', 'Wav2vec_Hidden_5', 'Wav2vec_Hidden_6',
                                   'Wav2vec_Hidden_7', 'Wav2vec_Hidden_8', 'Wav2vec_Hidden_9', 'Wav2vec_Hidden_10', 'Wav2vec_Hidden_11',
                                   'Wav2vec_Hidden_12', 'Wav2vec_Hidden_13', 'Ad_status', 'Depression_status', 'MMSE_score'])

        return df

    def data_preparation(self, feature_name: str, feature_data, original_df):
        if feature_name == "egemaps":
            index = [i for i, sublist in enumerate(feature_data) if any(math.isnan(x) for x in sublist)]
            for idx in sorted(index, reverse=True):
                del feature_data[idx]
                del original_df.Depression_status[idx]
                del original_df.Ad_status[idx]
            feature_data = np.array(feature_data.tolist())

        elif feature_name == "MFCC":
            feature_data = np.array(feature_data.tolist())
        else:
            feature_data = list(chain(*feature_data))
        return feature_data

    def preprocessing(self, original_df, feature_data, labels, random_seed, mode: str):
        """
        Preprocesses the input data for machine learning by splitting it into training and test sets,
        applying feature scaling, and optionally modifying target labels based on the specified mode.

        Parameters:
            self (object): The instance of the class containing this method.
            original_df (pandas.DataFrame): The original DataFrame containing the data.
            feature_data (pandas.DataFrame): The features used for prediction.
            labels (pandas.Series): The target labels for classification.
            random_seed (int): Random seed for reproducibility.
            mode (str): The mode for preprocessing. Must be one of ['Standard', 'Test_depression', 'Test_Ad'].

        Returns:
            X_train (numpy.ndarray): Scaled feature data for training.
            X_test (numpy.ndarray): Scaled feature data for testing.
            y_train (numpy.ndarray or pandas.Series): Target labels for training.
            y_test (numpy.ndarray or pandas.Series): Target labels for testing, possibly modified based on the mode.

        Notes:
            - When 'mode' is 'Standard', the function splits the data into training and test sets, scales the features,
              and returns them along with the original labels.
            - When 'mode' is 'Test_depression' or 'Test_Ad', the function additionally modifies the target labels based on
              the 'Depression_status' or 'Ad_status' column in 'original_df'.

        Example:
            X_train, X_test, y_train, y_test = preprocessing(self, original_df, feature_data, labels, random_seed, 'Standard')
        """

        if mode == 'Standard':
            X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3,
                                                                stratify=labels, random_state=random_seed)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        elif mode == 'Test_depression':
            X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3,
                                                                stratify=labels, random_state=random_seed)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            y_test = original_df.Depression_status.iloc[y_test.index]

        elif mode == 'Test_Ad':
            X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3,
                                                                stratify=labels, random_state=random_seed)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            y_test = original_df.Ad_status.iloc[y_test.index]

        return X_train, X_test, y_train, y_test

    def preprocessing_different_data(self, feature_data, labels, random_seed):
        X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3,
                                                            stratify=labels, random_state=random_seed)

        return X_train, y_train

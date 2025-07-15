"""
@author: Kashaf Gulzar

File to extract all the features and save them to pickle file
"""
# -------------- Import Modules ---------------
import os
from scipy.io import wavfile
import numpy as np
import acoustic_feature_functions
import pandas as pd
import metadata
import pickle

# -------------- Initilaizations ---------------
# Get the directory of the current script
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
results_directory = os.path.join(parent_directory, "Data", "Embeddings", "acoustic", "imbalanced_data_embedding.pkl")
data_dir = os.path.join(parent_directory, "Data", "Audio_data", "Segmented_audios")
metadata_dir = os.path.join(parent_directory, "Data", "Audio_data", "metadata_imbalanced.xlsx")
wav2vec_dir = os.path.join(parent_directory, "Data", "Wav2vec_features", "imbalanced_new")
met_data = pd.read_excel(metadata_dir)

feat = acoustic_feature_functions.GetFeatures()
final_embedding_mfcc = {}
final_embedding_mfcc_1d = {}
final_embedding_egemap = {}
final_embedding_latent_layer = {}
final_embedding_hidden_layer_1 = {}
final_embedding_hidden_layer_2 = {}
final_embedding_hidden_layer_3 = {}
final_embedding_hidden_layer_4 = {}
final_embedding_hidden_layer_5 = {}
final_embedding_hidden_layer_6 = {}
final_embedding_hidden_layer_7 = {}
final_embedding_hidden_layer_8 = {}
final_embedding_hidden_layer_9 = {}
final_embedding_hidden_layer_10 = {}
final_embedding_hidden_layer_11 = {}
final_embedding_hidden_layer_12 = {}
final_embedding_hidden_layer_13 = {}

# -------------- Feature extraction ---------------
"""
Get embeddings of MFCC, eGeMAPS and Wav2Vec and save it in a pickle file
"""
for root, directories, files in os.walk(data_dir):
    for filename in files:
        filepath = os.path.join(root, filename)
        fs, data = wavfile.read(filepath)
        data = np.float_(data)
        mfcc_embedding, mfcc_embedding_1d = feat.get_mfcc(data,fs)
        egemap_embedding = feat.get_egemaps(data,fs)
        final_embedding_mfcc[filename] = mfcc_embedding
        final_embedding_mfcc_1d[filename] = mfcc_embedding_1d
        final_embedding_egemap[filename] = egemap_embedding

for root, directories, files in os.walk(wav2vec_dir):
    for filename in files:
        filepath = os.path.join(root, filename)
        data = np.load(filepath, allow_pickle=True)
        final_embedding_latent_layer[filename] = data.tolist()["latent_layer"]
        final_embedding_hidden_layer_1[filename] = data.tolist()["hidden_layer_1"]
        final_embedding_hidden_layer_2[filename] = data.tolist()["hidden_layer_2"]
        final_embedding_hidden_layer_3[filename] = data.tolist()["hidden_layer_3"]
        final_embedding_hidden_layer_4[filename] = data.tolist()["hidden_layer_4"]
        final_embedding_hidden_layer_5[filename] = data.tolist()["hidden_layer_5"]
        final_embedding_hidden_layer_6[filename] = data.tolist()["hidden_layer_6"]
        final_embedding_hidden_layer_7[filename] = data.tolist()["hidden_layer_7"]
        final_embedding_hidden_layer_8[filename] = data.tolist()["hidden_layer_8"]
        final_embedding_hidden_layer_9[filename] = data.tolist()["hidden_layer_9"]
        final_embedding_hidden_layer_10[filename] = data.tolist()["hidden_layer_10"]
        final_embedding_hidden_layer_11[filename] = data.tolist()["hidden_layer_11"]
        final_embedding_hidden_layer_12[filename] = data.tolist()["hidden_layer_12"]
        final_embedding_hidden_layer_13[filename] = data.tolist()["hidden_layer_13"]

"""
Get test data dictionary with key=filename and value=test case object with embeddings along with metadeta
"""
test_data = {}

for f_name, embedding in final_embedding_mfcc.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_mfcc_matrix(embedding)

for f_name, embedding in final_embedding_mfcc_1d.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_mfcc_1d(embedding)

for f_name, embedding in final_embedding_egemap.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_egemaps(embedding)

for f_name, embedding in final_embedding_latent_layer.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_latentlayer(embedding)

for f_name, embedding in final_embedding_hidden_layer_1.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_1(embedding)

for f_name, embedding in final_embedding_hidden_layer_2.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_2(embedding)

for f_name, embedding in final_embedding_hidden_layer_3.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_3(embedding)

for f_name, embedding in final_embedding_hidden_layer_4.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_4(embedding)

for f_name, embedding in final_embedding_hidden_layer_5.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_5(embedding)

for f_name, embedding in final_embedding_hidden_layer_6.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_6(embedding)

for f_name, embedding in final_embedding_hidden_layer_7.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_7(embedding)

for f_name, embedding in final_embedding_hidden_layer_8.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_8(embedding)

for f_name, embedding in final_embedding_hidden_layer_9.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_9(embedding)

for f_name, embedding in final_embedding_hidden_layer_10.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_10(embedding)

for f_name, embedding in final_embedding_hidden_layer_11.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_11(embedding)

for f_name, embedding in final_embedding_hidden_layer_12.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_12(embedding)

for f_name, embedding in final_embedding_hidden_layer_13.items():
    f_name = f_name.split(".")[0]
    t_case = test_data.setdefault(f_name, metadata.TestData(f_name))
    t_case.set_hiddenlayer_13(embedding)


for index, series in met_data.iterrows():
    f_name, age, sex, mmse, group, depression, depression_score = series.values
    m_data = metadata.MetaData(f_name, age, sex, mmse, group, depression, depression_score)
    test_data[m_data.file].set_meta_data(m_data)

"""
Deletes files without meta data
"""
delete = [f_name for f_name, t_file in test_data.items() if t_file.metadata is None]
for f_name in delete:
    del test_data[f_name]
print(f"New dictionary: {test_data}")

"""
Stores features as a pickle file
"""
with open(results_directory, 'wb') as handle:
    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


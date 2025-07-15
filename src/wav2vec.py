
"""
Created on Sun Jan 30 15:45:14 2022

computes Wav2vec 2.0 embeddings from hidden layers
@author: Paula Perez
"""

import os
import warnings

warnings.filterwarnings("ignore")
import torch
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import gc
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector, Wav2Vec2ForPreTraining, Wav2Vec2Model, \
    Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor, Wav2Vec2Tokenizer

from transformers import pipeline
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from tqdm import tqdm
from scipy.io.wavfile import write, read
# from datasets import load_dataset
import numpy as np

torch.cuda.empty_cache()
gc.collect()
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForCTC, \
    Wav2Vec2Model
# from custom_feature_extraction_pipeline import CustomFeatureExtractionPipeline
# import soundfile as sf
# from datasets import load_dataset
def create_fold(new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)


# %%
def Wav2Vec_Models(file, plot=False, cuda=False):
    fs, audio = read(file)
    audio = (audio - np.mean(audio)) / max(audio)
    device = 'cpu'
    if cuda:
        device = 'cuda'

    w2v2_model="facebook/wav2vec2-base-960h"


    model = Wav2Vec2Model.from_pretrained(w2v2_model,output_hidden_states=True).to(device)

    feature_extractor = Wav2Vec2Processor.from_pretrained(w2v2_model)


    model.config.ctc_zero_infinity = True

    # audio file is decoded on the fly

    input_values = feature_extractor(np.array(audio, dtype=float), sampling_rate=fs, return_tensors="pt",
                                     padding=True).to(device).input_values
    input_values_all = feature_extractor(np.array(audio, dtype=float), sampling_rate=fs, return_tensors="pt",
                                         padding=True)


    # compute masked indices

    batch_size, raw_sequence_length = input_values.shape

    sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)

    mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)

    mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

    with torch.inference_mode():
        tmp = model(**input_values_all.to(device))
        
        outputs={}
        outputs['latent_layer']=np.array(torch.mean(tmp.extract_features, dim=1).cpu())
        j=0
        for vc in tmp.hidden_states:
            outputs['hidden_layer_'+str(j+1)]=np.array(torch.mean(vc, dim=1).cpu())
            j+=1
        
       

    return outputs








# %%
gc.collect()

tensors_ft = []
tensors_lh = []
labels = []
path = r'D:/MSc. AI/2nd semester/Alzheimers/Data/Audio_data/Segmented_audios/'
path_save = r'D:/MSc. AI/2nd semester/Alzheimers/Data/Wav2vec_features/unbalanced_new/'
#path_save = r'./Wav2vec_features/gender_ad_balanced/'
files = os.listdir(path)

create_fold(path_save)

pbar = tqdm(files)
for index, file in enumerate(pbar):
    pbar.set_description("Processing %s" % file)

    outputs = Wav2Vec_Models(path + file, cuda=False)

    np.save(path_save + file[:-4],outputs)
    torch.cuda.empty_cache()
    gc.collect()
gc.collect()

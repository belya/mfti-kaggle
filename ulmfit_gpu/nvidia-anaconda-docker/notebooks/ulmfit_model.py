from tqdm import tqdm_notebook
import torch
import fastai
from fastai.text import *
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load('comments.model')

class BPETokenizer(BaseTokenizer):
    def tokenizer(self, text):
        return sp.EncodeAsPieces(text)

def return_tokenizer(*args, **kwargs):
    return BPETokenizer(*args, **kwargs)

class ULMFiTModel:
    def load(self):
        self.data = load_data("./", "ulmfit_data_clas")
        self.classificator = text_classifier_learner(self.data, drop_mult=0.3, arch=AWD_LSTM)
        self.classificator = self.classificator.load("./comments_model_5")

    def preprocess(self, messages):
        return [message.lower() for message in messages]

    def predict_probabilities(self, messages):
        probabilities = [self.classificator.predict(item=message) for message in tqdm_notebook(messages)]
        return np.array([float(p[2][-1]) for p in probabilities])
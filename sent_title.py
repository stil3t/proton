import numpy as np
import pandas as pd
from dataclasses import dataclass

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

import os
from time import sleep
import argparse

import natasha
import razdel

emb = natasha.NewsEmbedding()

def embed(s):
    s = s.lower()
    
    while not (s in emb) and s:
        s = s[:-1]
        
    if not s:
        return emb['<unk>']
    else:
        return emb[s]
    
def vectorize_fixed(s, target_len=20):
    s = razdel.tokenize(s)
    res = np.zeros((target_len, 300))
    for i, word in enumerate(s):
        res[i] = embed(word.text)
    
    return res

def vectorize(s):
    s = razdel.tokenize(s)
    res = np.array([emb['<pad>']])
    for i, word in enumerate(s):
        res = np.vstack([res, embed(word.text)])
    
    return res[1:]

model = keras.models.load_model("/root/proton/my_model.keras")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title')
    args = parser.parse_args()
    sent = model.predict(vectorize_fixed(args.title, 130).reshape(1, 130, 300)).item()
    print(sent)
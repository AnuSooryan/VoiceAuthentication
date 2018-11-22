# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:34:20 2018

@author: Anu
"""

from SpeechDiarization.Features import mfcc
from sklearn.mixture import GaussianMixture 
from sklearn.externals import joblib
import os

path = os.getcwd() + '/SpeechDiarization/SpeechDiarization'

def model(login, file):#sign up 
    if file == '':
        features = mfcc(login, '')
        gmm = GaussianMixture(n_components = 8, max_iter = 200, 
                              covariance_type='diag', n_init = 3)
        gmm.fit(features)
        #dump 5 training samples #for likelihood score
        joblib.dump(gmm, path + '/speaker_models/' + login + '.pkl') 
        #dump mfcc features of 5 training samples# required for cosine similarity
#         joblib.dump(features, path + '/speaker_models/' + login + 'features.pkl')
    response = {}
    response['meassage'] = 'Model successfully built'
    return response
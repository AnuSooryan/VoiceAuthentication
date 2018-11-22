'''
Created on 24-Oct-2018

@author: anu
'''
from SpeechDiarization.Features import mfcc
from sklearn.mixture import GaussianMixture 
from sklearn.externals import joblib
import os

path = os.getcwd() + '/SpeechDiarization/SpeechDiarization'

def UBMmodel(ubmfoldername, file, login):
    if file == '':
        features = mfcc(ubmfoldername, '')
        gmm = GaussianMixture(n_components = 8, max_iter = 200, 
                              covariance_type='diag', n_init = 3)
        gmm.fit(features)
        joblib.dump(gmm, path + '/speaker_models/' + login + 'ubm.pkl')
        


'''
Created on 16-Oct-2018

@author: Anu
'''

from SpeechDiarization.Features import mfcc
from sklearn.externals import joblib
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize
import os
from math import ceil

path = os.getcwd() + '/SpeechDiarization/SpeechDiarization'
# svd = TruncatedSVD()

#login
def predict(login, file):
    login_features = mfcc(login, file)
    gmm = joblib.load(path + '/speaker_models/' + login + '.pkl')
    ubm = joblib.load(path + '/speaker_models/' + login + 'ubm.pkl')
#     signup_features = joblib.load(path + '/speaker_models/' + login + 'features.pkl')
    gmm_likelihood_score = gmm.score(login_features)#features of incoming voice
    ubm_likelihood_score = ubm.score(login_features)#features of incoming voice 
    likelihood_score = gmm_likelihood_score - ubm_likelihood_score
#     truncated_gmm = svd.fit_transform(signup_features)#svd #reducing multidimensional to 2D#signup
#     truncated_features = svd.fit_transform(login_features)#svd#login
#     features_array = truncated_features.ravel()
#     shape = features_array.shape[0]
#     gmm_array = truncated_gmm.ravel()[:shape]
#     similarity = cosine_similarity([gmm_array, features_array])
#     similarity = similarity[0][1]
    result = {}
#     result['similarity_score'] = similarity
    print(likelihood_score)
    if likelihood_score > 0:
        result['Message'] = 'Authenticated'
    else:
        result['Message'] = 'Not Authenticated'   
    return result
    
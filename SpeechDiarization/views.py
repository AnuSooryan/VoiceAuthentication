'''
Created on 09-Oct-2018

@author: Anu
'''
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import permission_classes
from rest_framework import permissions
from SpeechDiarization.SaveRecordings import folder, save, match_target_amplitude
from .prediction import predict
from SpeechDiarization.Model import model
from .ubm_model import UBMmodel
from pydub import AudioSegment
import io
import os 
import uuid
import shutil

path = os.getcwd() + '/SpeechDiarization/SpeechDiarization'

@permission_classes((permissions.AllowAny,))
class SaveFiles(viewsets.ViewSet):
    def create(self, request):
        files = request.FILES.getlist('file')
        login = request.POST['login_id']
        if len(files) == 5:
            Folder = folder(login)
            for f in files:
                saved = save(login, f)
            return Response(Folder)
        else:
            result = {}
            result['Message'] = 'Please input the fields'
            return Response(result)
   
@permission_classes((permissions.AllowAny,)) 
class ModelBuilding(viewsets.ViewSet):
    def create(self, request):
        question = request.data
        gmm = model(question['login_id'], question['file'])
        ubm = UBMmodel('UBM', '', question['login_id'])
        # deletion of folder after building gmm model #save storage
        shutil.rmtree(path + '/' + question['login_id'])
        current_path = path + '/' + 'UBM'
        file_list = os.listdir(current_path)
        f = [i for i in file_list if question['login_id'] in i]# to check filename loginid within ubm files
        for i in f:
            try:
                os.remove(current_path + '/' + i)
            except:
                pass
        return Response(gmm)
   
@permission_classes((permissions.AllowAny,))
class Predict(viewsets.ViewSet):
    def create(self, request):
        files = request.FILES.getlist('file')
        if len(files) > 0:
            file = files[0]
            filename = file.name
            format = filename.split('.')[-1:][0]
            login = request.POST['login_id']
            try:
                current_path = path + '/temp'
                for chunk in file.chunks():
                    Audio = AudioSegment.from_file(io.BytesIO(chunk), format=format)
                    normalized_audio = match_target_amplitude(Audio, -20.0)
                    file_path = os.path.join(current_path, login + uuid.uuid4().hex) 
                    normalized_audio.export(file_path+'.wav', format="wav")
                file_name = file_path + '.wav'
                prediction = predict(login, file_name)
                if os.path.exists(file_name):
                    os.remove(file_name)
                return Response(prediction)
            except:
                result = {}
                result['Message'] = 'Please sign up'
                if os.path.exists(file_name):
                    os.remove(file_name)
                return Response(result)
        else:
            result = {}
            result['Message'] = 'Please enter your voice note' 
            return Response(result)


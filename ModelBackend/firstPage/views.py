from django.shortcuts import render
from django.http import JsonResponse
# Create your views here.
import json
import pandas as pd
from django.core.files.storage import FileSystemStorage
import joblib

model=joblib.load('modelPipeline.pkl')



def scoreJson(request):
    data =json.loads(request.body)
    dataF=pd.DataFrame(data,index=[0])
    score=model.predict_proba(dataF)[:,-1][0]
    score=float(score)
    print(score)
    return JsonResponse({'score':1})


def scoreFile(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    filePath='.'+filePathName
    data =pd.read_csv(filePath)
    data2=data.columns[1:]
    score=model.predict_proba(data[data2])[:,-1]
    print(data[data2])
    # score={j:k for j,k in zip(data['Loan_ID'],score)}

    # score =sorted(score.items(),key=lambda x: x[1],reverse=True)
    score.sort()
    return JsonResponse({'result':1})
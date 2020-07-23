import os
import pandas as pd
import numpy as np
import io
from google.colab import files
url="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/train.csv"
UrlTest="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/test.csv"
UrlSubmission="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/sample_submission.csv"
#uploaded = files.upload()
#df2 = pd.read_csv(io.BytesIO(uploaded['train.csv']))
#df2.head()
def read_train():
  train = pd.read_csv(url)
  train['text']=train['text'].astype(str)
  train['selected_text']=train['selected_text'].astype(str)
  return train
def read_test():
    test=pd.read_csv(UrlTest)
    test['text']=test['text'].astype(str)
    return test
def read_submission():
    test=pd.read_csv(UrlSubmission)
    return test
train_df = read_train()
test_df = read_test()
submission_df = read_submission()


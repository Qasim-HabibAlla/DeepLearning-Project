import os
import pandas as pd
import numpy as np
import io
import re
from google.colab import files
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import string
import json
!pip install torch torchvision
!pip install transformers==2.2.0
!pip install seqeval
!pip install tensorboardx
!pip install simpletransformers==0.9.1
from simpletransformers.question_answering import QuestionAnsweringModel
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from google.colab import drive
drive.mount('/content/gdrive')
stop=set(stopwords.words('english'))
url="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/train.csv"
UrlTest="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/test.csv"
UrlSubmission="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/sample_submission.csv"

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
sub_df = read_submission()
def remove_stopwords(text):
        if text is not None:
            tokens = [x for x in word_tokenize(text) if x not in stop]
            return " ".join(tokens)
        else:
            return None
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
def remove_punct(text):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in text if ch not in exclude)
    return s
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def clean_df(df, train=True):
    df["dirty_text"] = df['text']
    
    
    df["text"] = df['text'].apply(lambda x : x.lower())
    
    df['text']=df['text'].apply(lambda x: remove_emoji(x))
        
    df['text']=df['text'].apply(lambda x : remove_URL(x))
        
    df['text']=df['text'].apply(lambda x : remove_html(x))
    df['text'] =df['text'].apply(lambda x : remove_stopwords(x)) 
    
    df['text']=df['text'].apply(lambda x : remove_punct(x))
    
    df.text = df.text.replace('\s+', ' ', regex=True)
    if train:
        df["selected_text"] = df['selected_text'].apply(lambda x : x.lower())
        df['selected_text']=df['selected_text'].apply(lambda x: remove_emoji(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_URL(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_html(x))
        df['selected_text'] =df['selected_text'].apply(lambda x : remove_stopwords(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_punct(x))
        df.selected_text = df.selected_text.replace('\s+', ' ', regex=True)
    
    return df
#train_df = clean_df(train_df)
#test_df = clean_df(test_df, train=False)
train = np.array(train_df)
test = np.array(test_df)
!mkdir -p data

use_cuda = True
def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1
def do_qa_train(train):

    output = []
    for line in train:
        context = line[1]

        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer.lower()})
            break
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})

        output.append({'context': context.lower(), 'qas': qas})
        
    return output
qa_train = do_qa_train(train)
with open('/content/gdrive/My Drive/data/train.json', 'w') as outfile:
    json.dump(qa_train, outfile)
def do_qa_test(test):
    output = []
    for line in test:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answers = []
        answers.append({'answer_start': 1000000, 'text': '__None__'})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
        output.append({'context': context.lower(), 'qas': qas})
    return output
qa_test = do_qa_test(test)
with open('/content/gdrive/My Drive/data/test.json', 'w') as outfile:
    json.dump(qa_test, outfile)
MODEL_PATH = '/content/gdrive/My Drive/model_deeplearning/'
#MODEL_PATH = 'https://drive.google.com/drive/folders/1CkjjRb6GJENfPQqfDJgVnzwipShmy4RE?usp=sharing'
model = QuestionAnsweringModel('distilbert', 
                               MODEL_PATH, 
                               args={'reprocess_input_data': True,
                                     'overwrite_output_dir': True,
                                     'learning_rate': 5e-5,
                                     'num_train_epochs': 3,
                                     'max_seq_length': 192,
                                     'doc_stride': 64,
                                     'fp16': False,
                                    },
                              use_cuda=True)
model.train_model('/content/gdrive/My Drive/data/train.json')
predictions = model.predict(qa_test)
predictions_df = pd.DataFrame.from_dict(predictions)

sub_df['selected_text'] = predictions_df['answer']

sub_df.to_csv('/content/gdrive/My Drive/sample_submission.csv', index=False)

print("File submitted successfully.")
#test_df.head()


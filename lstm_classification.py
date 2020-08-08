import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from google.colab import files
import os
url="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/train.csv"
UrlTest="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/test.csv"
UrlSubmission="https://raw.githubusercontent.com/Qasim-HabibAlla/deeplearninigdata/master/sample_submission.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1000)
#df2 = pd.read_csv(io.BytesIO(uploaded['train.csv']))
#df2.head()
def read_train():
  train = pd.read_csv(url)
  train['text']=train['text'].astype(str)
  train['sentiment']=train['sentiment'].astype(str)
  train['selected_text']=train['selected_text'].astype(str)
  return train
def read_test():
    test=pd.read_csv(UrlTest)
    test['text']=test['text'].astype(str)
    test['sentiment']=test['sentiment'].astype(str)
    return test
def read_submission():
    test=pd.read_csv(UrlSubmission)
    return test
train_df = read_train()
test_df = read_test()
sub_df = read_submission()
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
def remove_punct(text):
    exclude = set(string.punctuation)
    s=''
    for ch in text:
      if ch in exclude:
        s+=' '
      else:
        s+=ch
    #s = ''.join(ch for ch in text if ch not in exclude)
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
   # df['text'] =df['text'].apply(lambda x : remove_stopwords(x)) 
    
    df['text']=df['text'].apply(lambda x : remove_punct(x))
    
    df.text = df.text.replace('\s+', ' ', regex=True)
    if train:
        df["selected_text"] = df['selected_text'].apply(lambda x : x.lower())
        df['selected_text']=df['selected_text'].apply(lambda x: remove_emoji(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_URL(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_html(x))
       # df['selected_text'] =df['selected_text'].apply(lambda x : remove_stopwords(x))
        df['selected_text']=df['selected_text'].apply(lambda x : remove_punct(x))
        df.selected_text = df.selected_text.replace('\s+', ' ', regex=True)
    
    return df
train_df = clean_df(train_df)
test_df = clean_df(test_df, train=False)
train = np.array(train_df)
test = np.array(test_df)
train[18][2]='gonna miss every one'
train[251][2]='powerblog what is this powerblog challenge you keep talking about im a newbie follower'
train[458][2]='wow what a beautiful picture'
train[309][2]='it was worth a shot though'
train[492][2]='sorry'
train[569][2]='i dont think ive ever been so tierd in my life'
train[581][2]='no bueno hollykins needs to feel better asap ps i miss you you done with uni soon arent you soproudofyou'
train[639][2]='oooo ok why havent you accepted my friends request'
train[637][2]='hi to one kiwi artist from another kiwi artist'
train[757][2]='hes amazing'
train[787][2]='flew brisbane lax today great flight love the lights shame about one drink limit though'
train[1039][2]='yes now would be good'
train[1077][2]='it looks amazing'
train[1264][1]='ray bright eyes contentment'
train[1264][2]='contentment'
train[1319][1]='shoesshoesshoes yayyayyay lol iwouldpostatwitpic butidntknohow2'
train[1319][2]='shoesshoesshoes yayyayyay lol'
train[1696][1]='face and nana wish i was there last night'
train[1696][2]='wish'
train[1728][2]='lovely'
train[1754][2]='i can t has blocked me i can t even request'
train[1798][2]='happy'
train[2187][2]='failed on first day twice'
train[2787][2]='could motivate'
train[2986][2]='be useful'
train[3369][2]='we enjoyed'
train[3400][2]='wow i hope he gets better'
train[4654][2]='have a great summer'
train[4747][2]='splendid evening'
train[5189][1]='cracking myself more more up phootoboothingis fun for bunny volumen eins'
train[5189][2]='fun'
train[5196][2]='very kewl'
train[5358][2]='terrible'
train[5530][2]='welcome'
train[5560][2]='sorries'
train[5687][2]='why we canï¿½t'
train[5712][2]='good morning all hope everyone is doing well on this monday thanks for all the followfriday recos i am blessed'
train[6113][2]='these dogs are going to die if somebody doesn t save them '
train[6131][2]='so tired'
train[9535][2]='i don t feel so good'
train[17404][2]='feelin nice'
train[6230][2]='overwhelming'
train[6528][2]='great'
train[19963][2]='have a great day'
train[6686][2]='have a wonderful day'
train[25712][2]='a wonderful'
train[6724][2]='soooo romantic even'
train[7040][2]='save even'
train[13796][2]='happy mother s day everyone'
train[19162][2]='why are paracetamol so hard to swallow even'
train[6948][2]='seriously condolences'
train[7024][2]='happy mothers day to everyone'
train[7409][2]='wow ur page is awesome'
train[15165][2]='hehe i will never thorw out these shoes i m listening to varsity fanclub surprise surprise sway sway baby is awesome'
train[26625][2]='an awesome'
train[7642][2]='be painful'
train[8049][2]='smile life is good'
train[8153][2]='not good'
train[21876][2]='obnoxiously closer'
train[8235][2]='is literally amazing'
train[8249][2]='hi i m ok still not feeling great'
train[12356][2]='was great'
train[15010][2]='great musical'
train[23617][2]='great'
train[24766][2]='2 great'
train[26256][2]='a great'
train[8706][2]='it s good isn our'
train[19028][2]='just chilling out'
train[19057][2]='hell outside'
train[24996][2]='deserve so much better'
train[8803][2]='good morning fella i have the joy of work to do today'
train[9113][2]='very cute'
train[27121][2]='very nervous'
train[9539][2]='richgirl i m a sucker for the later'
train[9594][2]='neways im chillin'
train[10007][2]='time to leave a passive agressive note to the owners it s not the dog s fault it s their owners'
train[10492][2]='i m smiling'
train[10672][2]='pride and prejudice emma are great'
train[10968][2]='probably better'
train[10981][2]='they re gorgeous'
train[11353][2]='i m frustraded'
train[11480][2]='the best'
train[22864][2]='the best'
train[11698][2]='goodnite'
train[11706][2]='have the sweetest children you are obviously a great mom i loved reading your blog w their msgs to you kudos'
train[11745][2]='shortcakefail'
train[12138][2]='weird'
train[12205][2]='i wish'
train[12416][2]='was clean'
train[12522][2]='goodbye'
train[12576][2]='was looking forward'
train[12662][2]='but don t worry'
train[12736][2]='omg i just slept like 18hrs in the last 22hrs i think i m dying or something reminds me of catcher'
train[12803][2]='oh i m so happy'
train[13803][2]='happy'
train[13907][2]='happy'
train[14571][2]='get back happy'
train[24753][2]='happy'
train[26927][2]='happy'
train[13365][2]='looks super sweet'
train[13704][2]='god peppermint mochas frappachinos are amazing and addicting'
train[14213][2]='i enjoy'
train[22205][2]='you enjoy'
train[14257][2]='sorry'
train[14275][2]='i missed'
train[25499][2]='missed'
train[14611][2]='so awesome'
train[14855][2]='good morning to all and welcome new'
train[16372][2]='so different'
train[16643][2]='it was sooo awesome'
train[22365][2]='is awesome'
train[16720][2]='happy mom s get to rule'
train[16726][2]='bastards'
train[17062][2]='sorry'
train[22387][2]='sorry'
train[23680][2]='sorry'
train[26268][2]='sorry'
train[27362][2]='twitpic sorry'
train[17513][2]='aww thanks glad i appreciated'
train[27386][2]='much appreciate'
train[17627][2]='you sheesh'
train[17762][2]='it s depressing'
train[18003][2]='they are all amazing people'
train[18314][2]='good guy'
train[18862][2]='i liked their manly man ness'
train[19213][2]='hope they get you in soon and can make you feel all better'
train[19239][2]='though i liked them but i m also crazy'
train[19562][2]='charmingly funny'
train[19634][2]='good lord man i recommend'
train[19754][2]='cancelling my javaone'
train[20299][2]='darn'
train[20506][2]='glorious sundays'
train[20865][2]='i m quite excited'
train[20895][2]='my phone is broken im too lazy'
train[21205][2]='watching everyone else act a fool is much better the not remembering acting a fool yourself'
train[21755][2]='can t tell you how thrilled i am to have just had a nosebleed first time in ages overjoyed doesn t cover it urgh'
train[22234][2]='is ignoring'
train[22536][2]='metsies fan also boo hoo'
train[22769][2]='dammit hulu'
train[23081][2]='well doesnt deserve'
train[23199][2]='say i still miss waking up to your pleasant personality'
train[23290][2]='are delicious'
train[23733][2]='there is always tomorrow'
train[23746][2]='just harder'
train[24026][2]='just joking'
train[24490][2]='gonna b nervvoouuss'
train[25104][2]='just ignore'
train[25446][2]='ohhh that cant be very fun but hell you manned it up looks like you got some done today'
train[25760][2]='girls love u guys smiles'
train[26643][2]='one day my hugs will come fingers still crossed'
train[26677][2]='jus glad'
train[26830][2]='bracelet broke'
train[27209][2]='devices failed'
train[27229][2]='can t wait to see the smile on her face'
train[27280][2]='i can t move either'
train[27401][2]='jaja enjoyyitverymuch'
for i,line in enumerate(train):
  if line[3]=='neutral':
    train[i][2]=train[i][1]
found=0
SOS_token = 0
EOS_token = 1
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def CreatePairs(input_data,TypeData):
  pairs=[]
  if TypeData=='train':
    num_index=3
  else:
    num_index=2
  for line in input_data:
    input_tweet=line[1]
    output_tweet=line[num_index]
    if output_tweet=='neutral':
      output_tweet=0
    elif output_tweet=='negative':
      output_tweet=1
    elif output_tweet=='positive':
      output_tweet=2
    pairs.append((input_tweet,output_tweet))
  return pairs
def PrepareData(data,TypeData):
  pairs=CreatePairs(data,TypeData)
  for pair in pairs:
        input_lang.addSentence(pair[0])
  return pairs
input_lang = Lang('tweet')
pairs_training = PrepareData(train,'train')
pairs_test = PrepareData(test,'test')
MAX_LENGTH = 0
for line in test:
  if len(line[1])>MAX_LENGTH:
    MAX_LENGTH=len(line[1])
for line in train:
  if len(line[1])>MAX_LENGTH:
    MAX_LENGTH=len(line[1])
class RNNModule(nn.Module):
  def __init__(self, input_size, hidden_size,output_size=3):
    super(RNNModule, self).__init__()
    self.hidden_size = hidden_size
    self.hidden_var = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.net = nn.LSTM(hidden_size,self.hidden_var)
    self.linear = nn.Linear(self.hidden_var, output_size)
  def forward(self, input, hidden):
    embedded = self.embedding(input).view(len(input), 1, -1)
    output, hidden = self.net(embedded, hidden)
    predictions = self.linear(output.view(len(input), -1))
    predictions= F.log_softmax(predictions,dim=1)
    predictions=predictions[-1]
    return predictions.view(1, 3)
  def initHidden(self):
     h_state = torch.zeros(1, 1, self.hidden_var, device=device)
     c_state = torch.zeros(1, 1, self.hidden_var, device=device)
     hidden = (h_state, c_state)
     return hidden

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor=torch.tensor(pair[1], dtype=torch.long, device=device).view(-1, 1)
    #target_tensor = tensorFromSentence(input_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    loss = 0
    encoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    output_prdiction=encoder(input_tensor, encoder_hidden)
    loss += criterion(output_prdiction, target_tensor[0])
    loss.backward()
    encoder_optimizer.step()
    return loss.item()
training_pairs=[]
test_pairs=[]
for pair in pairs_training:
  training_pairs.append(tensorsFromPair(pair))
for pair in pairs_test:
  test_pairs.append(tensorsFromPair(pair))
def trainIters(encoder, learning_rate=0.01,print_every=1000,epochs=15):
  print_loss_total = 0
  encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()
  for e in range(epochs):
    print('*********************')
    print('epochs number:',e)
    for iter in range(len(training_pairs)):
      training_pair = training_pairs[iter]
      input_tensor = training_pair[0]
      target_tensor = training_pair[1]
      loss = train(input_tensor, target_tensor, encoder,encoder_optimizer, criterion)
      print_loss_total += loss
      if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('iter number: ',iter,'the average loss is :',print_loss_avg)
    evaluate(encoder)
def evaluate(encoder, max_length=MAX_LENGTH):
  correct_count=0 
  for i,test_input in enumerate(test_pairs):
    if i%100==0:
      print('test number:', i)
    with torch.no_grad():
      input_tensor = test_input[0]
      target_tensor=test_input[1]
      input_length = input_tensor.size()[0]
      encoder_hidden = encoder.initHidden()
      encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
      output_prdiction=encoder(input_tensor, encoder_hidden)
      ps = torch.exp(output_prdiction)
      probab = list(ps.cpu().numpy())
      max_index=0
      max_prob=0
      for i,prob in enumerate(probab[0]):
        if prob>=max_prob:
          max_index=i
          max_prob=prob
      true_label = target_tensor.cpu().numpy()
      if(true_label == max_index):
        correct_count+=1
  print("\nModel Accuracy of test set ={}%".format(int(100*(correct_count/len(test_pairs)))))

hidden_size = 256
encoder1 = RNNModule(input_lang.n_words, hidden_size).to(device)
trainIters(encoder1)
evaluate(encoder1)
                      
import nltk,numpy,keras
from keras.preprocessing.text import *
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import Dense, Flatten ,LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from numpy import array

numpy.random.seed(7)
import pandas as pd
stop = stopwords.words('english')

#read News List from Excel
df = pd.read_excel(r'Corpus.xlsx',header=0)

df=df.drop_duplicates()

def preprocess(corpus):
	nres=[]
	word_length = 0
	for i in range(len(corpus)):
		#Get location of News Corpus
		para=str(corpus[i][0])
		#Get Tokens of Paragraph
		word_tokens = word_tokenize(para)
		#Use Keras library to remove filters and make lowercase
		words = set(text_to_word_sequence(word_tokens))
		#Remove stopwords
		filtered_sentence = [w for w in words if not w in stop]
		#Get word_tokens after removing stopwords
		word_length = word_length + len(filtered_sentence)
		para1=" ".join(x for x in filtered_sentence)
		print('length '+str(len(para1)))
		vocab_size = len(words)
		print(vocab_size)
		# integer encode the document
		result = one_hot(para1, round(vocab_size*1.3))
		print(result)
		nres.append(result)
	return nres
    
print('done')
max_length = 400
#This pad is to make every vector of same length
train_vector=preprocess(df[corpus][:400])
padded_dos = sequence.pad_sequences(train_vector,maxlen = 400,padding='post')
from numpy import array
df['Sentiment'].replace({1:0,2:1,3:2},inplace=True)
label = df['Sentiment'].tolist()
label = to_categorical(label)
label = array(label)


model = Sequential()
model.add(Embedding(input_dim=word_length,output_dim=4,input_length=400))
#model.add(Flatten())
model.add(LSTM(10, input_dim = word_length, input_length =400 , return_sequences=True))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(3, return_sequences=True))
#model.add(Dense(units=10,activation='relu'))
#model.add(Dense(units=8,activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
print(model.summary())

model.fit(padded_dos,label,epochs=40,batch_size=32)

scores = model.evaluate(padded_dos, label, verbose=0)
print("Training Accuracy: %.2f%%" % (scores[1]*100))

#Testing
test_vector=preprocess(df[corpus][401:600])
test_padded_docs = sequence.pad_sequences(test_vector,maxlen = 400,padding='post')
y_pred = model.predict(test_padded_docs)
classes = y_pred.argmax(axis=-1) 

count=0
for i in range(20):
    if(test['Sentiment'][:20][i]==classes[:20][i]):
        count+=1
per=count/20
print("Test Accuracy: %.2f%%" % (per*100))


test['Predicted_Output'] = classes
test.to_excel('Output_Test.xlsx',index=False)
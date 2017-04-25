import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
import nltk as nt
import time
import datetime

def one_hot(pos, dim):
    oh = np.zeros(dim)
    oh[pos] = 1
    return oh


def get_id(pos, dic):
    if pos not in dic:
        dic[pos] = len(dic)
    return dic[pos]

ts = time.time()
print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
pos_tags = {}

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
features = pd.read_csv('data/quora_features.csv')

common_words = features.common_words.values
common_words_dim = max(common_words) + 1

print type(common_words)
print common_words.shape

# common_words_oh = features.common_words.apply(lambda x: one_hot(x, common_words_dim))
# for i in common_words:
#     common_words_oh.append(one_hot(i, common_words_dim))

# print type(common_words_oh)
# print common_words_oh

# common_words_reshaped = np.reshape(common_words_oh, (common_words_oh.shape[0], common_words_dim))
# print common_words_reshaped.shape
# print common_words_reshaped

# model7 = Sequential()
# model7.add(LSTM(common_words_dim, input_dim=common_words_dim, dropout_W=0.2, dropout_U=0.2))
# model7 = Sequential()
# model7.add(Embedding(common_words_dim, common_words_dim, input_length=common_words_dim))
# model7.add(LSTM(common_words_dim, dropout_W=0.2, dropout_U=0.2))


ts = time.time()
print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

y = data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

print type(x1)
print x1.shape

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

q1_pos_tags = []
for q in data.question1:
    #print q
    try:
    	question = nt.word_tokenize(q)
    	tags = nt.pos_tag(question)
    	current_posvec = []
    	for i in range(len(tags)):
        	#postagid = get_id(tags[i][1],pos_tags)
		current_posvec.append(get_id(tags[i][1],pos_tags))
	q1_pos_tags.append(current_posvec)
    except:
        q1_pos_tags.append([0])
q1_pos_tags = sequence.pad_sequences(q1_pos_tags, maxlen=max_len)
print q1_pos_tags

ts = time.time()
print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

q2_pos_tags = []
for q in data.question2:
    #print q
    try:
    	question = nt.word_tokenize(q)
    	tags = nt.pos_tag(question)
    	current_posvec = []
    	for i in range(len(tags)):
        	#postagid = get_id(tags[i][1],pos_tags)
		current_posvec.append(get_id(tags[i][1],pos_tags))
	q2_pos_tags.append(current_posvec)
    except:
        q2_pos_tags.append([0])
q2_pos_tags = sequence.pad_sequences(q2_pos_tags, maxlen=max_len)

print q2_pos_tags
ts = time.time()
print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

word_index = tk.word_index


ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

model7 = Sequential()
model7.add(Embedding(common_words_dim, 10, input_length=1))
model7.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

model8 = Sequential()
model8.add(Embedding(len(pos_tags)+1, 10, input_length=max_len))
model8.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

model9 = Sequential()
model9.add(Embedding(len(pos_tags)+1, 10, input_length=max_len))
model9.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6, model7, model8, model9], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

merged_model.fit([x1, x2, x1, x2, x1, x2, common_words, q1_pos_tags, q2_pos_tags], y=y, batch_size=384, nb_epoch=2,
                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

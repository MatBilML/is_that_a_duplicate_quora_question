import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge, Flatten,Permute, RepeatVector, merge, Input
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
import nltk as nt
import time
import datetime
import sys
import optparse
from keras.regularizers import l1, l2
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Helper functions

def one_hot(pos, dim):
    oh = np.zeros(dim)
    oh[pos] = 1
    return oh


def get_id(pos, dic):
    if pos not in dic:
        dic[pos] = len(dic)
    return dic[pos]


def print_current_time():
    print get_current_time()


def get_current_time():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' '


def get_pos_tag_sequence_padded(questions, max_length):
    q_pos_tags = []
    for question in questions:
        try:
            tokens = nt.word_tokenize(question)
            tags = nt.pos_tag(tokens)
            current_posvec = []
            for index in range(len(tags)):
                current_posvec.append(get_id(tags[index][1], pos_tags))
            q_pos_tags.append(current_posvec)
        except:
            q_pos_tags.append([0])
    q_pos_tags = sequence.pad_sequences(q_pos_tags, maxlen=max_length)
    return q_pos_tags


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-b', '--baseline', action='store',
                         type='int', dest='baseline', default=0,
                         help='Run baseline version?')
    optParser.add_option('-a', '--attention', action='store',
                         type='int', dest='attention', default=0,
                         help='Include attention in the model?')
    optParser.add_option('-c', '--cnn', action='store',
                         type='int', dest='cnn', default=1,
                         help='add cnn before LSTM?')
    optParser.add_option('-r', '--regularization', action='store',
                         type='int', dest='regularize', default=0,
                         help='L2 regularization in LSTM')
    optParser.add_option('-l', '--bilstm', action='store',
                         type='int', dest='bilstm', default=1,
                         help='Use bilstm')
    optParser.add_option('-p', '--postags', action='store',
                         type='int', dest='postags', default=0,
                         help='Include postags in the model')
    optParser.add_option('-e', '--epochs', action='store',
                         type='int', dest='epochs', default=2,
                         help='Number of epochs')

    opts, args = optParser.parse_args()
    return opts


# Main execution
opts = parseOptions()
print opts
NUM_EPOCHS = opts.epochs
if opts.baseline == 0:
    print get_current_time(), 'Running latest version.'
else:
    print get_current_time(), 'Running baseline version.'

print get_current_time(), 'Starting execution . . .'
pos_tags = {}

data = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
features = pd.read_csv('data/quora_features.csv')

common_words = features.common_words.values
common_words_dim = max(common_words) + 1

print get_current_time(), 'type(common_words): ', type(common_words)
print get_current_time(), 'common_words.shape: ', common_words.shape

y = data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

print get_current_time(), 'type(x1): ', type(x1)
print get_current_time(), 'x1.shape: ', x1.shape

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

if opts.postags==1:
    print get_current_time(), 'Getting POS tags for q1 set . . .'
    q1_pos_tags = get_pos_tag_sequence_padded(data.question1, max_len)
    print get_current_time(), 'q1_pos_tags: ', q1_pos_tags

    print get_current_time(), 'Getting POS tags for q2 set . . .'
    q2_pos_tags = get_pos_tag_sequence_padded(data.question2, max_len)
    print get_current_time(), 'q2_pos_tags: ', q2_pos_tags

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

print get_current_time(), 'Generating embedding index . . .'
embeddings_index = {}
f = open('data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print(get_current_time() + 'Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print get_current_time(), 'Embedding matrix generated . . .'

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

print(get_current_time() + 'Building model...')

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

#model5
if opts.attention == 1:
    model5_ip = Input(shape=(40,))
    x5 = Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2)(model5_ip)
    if opts.cnn == 1:
        x5 = Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x5)
        x5 = MaxPooling1D(pool_size=4)(x5)
    if opts.bilstm == 1:
        if opts.regularize == 1:
            x5 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True, W_regularizer=l2(0.01)))(
                x5)
        else:
            x5 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(x5)
    else:
        if opts.regularize == 1:
            x5 = LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True, W_regularizer=l2(0.01))(
                x5)
        else:
            x5 = LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(x5)


    attention5 = TimeDistributed(Dense(1, activation='tanh'))(x5)
    attention5 = Flatten()(attention5)
    attention5 = Activation('softmax')(attention5)
    attention5 = RepeatVector(600)(attention5)
    attention5 = Permute([2, 1])(attention5)

    merge5 = merge([x5, attention5], mode='mul')
    merge5 = Lambda(lambda xin: K.sum(xin, axis=1))(merge5)
    merge5 = Dense(300, activation='softmax')(merge5)

    model5 = Model(input=model5_ip, output=merge5)
    print model5.summary()

else:
    model5 = Sequential()
    model5.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    if opts.cnn == 1:
        model5.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
        model5.add(MaxPooling1D(pool_size=4))
    if opts.bilstm == 1:
        if opts.regularize == 1:
            model5.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, W_regularizer=l2(0.01))))
        else:
            model5.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2)))
    else:
        if opts.regularize == 1:
            model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, W_regularizer=l2(0.01)))
        else:
            model5.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))
    print model5.summary()

# model6
if opts.attention == 1:
    model6_ip = Input(shape=(40,))
    x6 = Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2)(model6_ip)
    if opts.cnn == 1:
        x6 = Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x6)
        x6 = MaxPooling1D(pool_size=4)(x6)
    if opts.bilstm == 1:
        if opts.regularize == 1:
            x6 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True, W_regularizer=l2(0.01)))(
                x6)
        else:
            x6 = Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True))(x6)
    else:
        if opts.regularize == 1:
            x6 = LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True, W_regularizer=l2(0.01))(
                x6)
        else:
            x6 = LSTM(300, dropout_W=0.2, dropout_U=0.2, return_sequences=True)(x6)

    attention6 = TimeDistributed(Dense(1, activation='tanh'))(x6)
    attention6 = Flatten()(attention6)
    attention6 = Activation('softmax')(attention6)
    attention6 = RepeatVector(600)(attention6)
    attention6 = Permute([2, 1])(attention6)

    merge6 = merge([x6, attention6], mode='mul')
    merge6 = Lambda(lambda xin: K.sum(xin, axis=1))(merge6)
    merge6 = Dense(300, activation='softmax')(merge6)

    model6 = Model(input=model6_ip, output=merge6)
    print model6.summary()

else:
    model6 = Sequential()
    model6.add(Embedding(len(word_index) + 1, 300, input_length=40, dropout=0.2))
    if opts.cnn == 1:
        model6.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
        model6.add(MaxPooling1D(pool_size=4))
    if opts.bilstm == 1:
        if opts.regularize == 1:
            model6.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2, W_regularizer=l2(0.01))))
        else:
            model6.add(Bidirectional(LSTM(300, dropout_W=0.2, dropout_U=0.2)))
    else:
        if opts.regularize == 1:
            model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2, W_regularizer=l2(0.01)))
        else:
            model6.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))
    print model6.summary()

model7 = Sequential()
model7.add(Embedding(common_words_dim, 10, input_length=1))
model7.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

if opts.postags:
    model8 = Sequential()
    model8.add(Embedding(len(pos_tags) + 1, 10, input_length=max_len))
    model8.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    model9 = Sequential()
    model9.add(Embedding(len(pos_tags) + 1, 10, input_length=max_len))
    model9.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

merged_model = Sequential()
if opts.baseline == 1:
    merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
else:
    if opts.postags == 1:
        merged_model.add(Merge([model1, model2, model3, model4, model5, model6, model7, model8, model9], mode='concat'))
    else:
        merged_model.add(Merge([model1, model2, model3, model4, model5, model6, model7], mode='concat'))

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

result = []
if opts.baseline == 1:
    result = merged_model.fit([x1, x2, x1, x2, x1, x2], y=y, batch_size=384, nb_epoch=NUM_EPOCHS,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
else:
    if opts.postags == 1:
        result = merged_model.fit([x1, x2, x1, x2, x1, x2, common_words, q1_pos_tags, q2_pos_tags], y=y, batch_size=384, nb_epoch=NUM_EPOCHS,
                     verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
    else:
        result = merged_model.fit([x1, x2, x1, x2, x1, x2, common_words], y=y, batch_size=384,
                         nb_epoch=NUM_EPOCHS,
                         verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

print(result.history.keys())
#plot Accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.png')
plt.clf()

#Plot loss
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.png')
plt.clf()
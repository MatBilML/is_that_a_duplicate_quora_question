import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge, Flatten, Permute, RepeatVector, merge, Input
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
import os
import optparse
from keras.regularizers import l1, l2
from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ast import literal_eval
import math

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


def get_pos_tag_sequence_padded(pos_tag_features, max_length):
    q_pos_tags = []
    for pos_tag_feature in pos_tag_features:
        current_posvec = []
        pos_tag_feature = literal_eval(pos_tag_feature)
        for index in range(len(pos_tag_feature)):
            current_posvec.append(get_id(pos_tag_feature[index][1], pos_tags))
        q_pos_tags.append(current_posvec)
    q_pos_tags = sequence.pad_sequences(q_pos_tags, maxlen=max_length)
    return q_pos_tags


def get_chunk_tag_sequence_padded(chunk_features, max_length):
    q_chunk_tags = []
    for chunk_feature in chunk_features:
        current_chunkvec = []
        chunk_feature = literal_eval(chunk_feature)
        for index in range(len(chunk_feature)):
            current_chunkvec.append(get_id(chunk_feature[index][1], chunk_tags))
        q_chunk_tags.append(current_chunkvec)
    q_chunk_tags = sequence.pad_sequences(q_chunk_tags, maxlen=max_length)
    return q_chunk_tags

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def get_srl_tag_sequence_padded(srl_tag_features, max_length):
    q_srl_tags = []
    for srl_tag_feature in srl_tag_features:
        current_srlvec = []
        srl_tag_feature = literal_eval(srl_tag_feature)
        for srl_tag in srl_tag_feature:
            for key in srl_tag.keys():
                current_srlvec.append(get_id(key, srl_tags))
        q_srl_tags.append(current_srlvec)
    q_srl_tags = sequence.pad_sequences(q_srl_tags, maxlen=max_length)
    return q_srl_tags


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-b', '--baseline', action='store',
                         type='int', dest='baseline', default=0,
                         help='Run baseline version?')
    optParser.add_option('-a', '--attention', action='store',
                         type='int', dest='attention', default=0,
                         help='Include attention in the model?')
    optParser.add_option('-c', '--cnn', action='store',
                         type='int', dest='cnn', default=0,
                         help='add cnn before LSTM?')
    optParser.add_option('-r', '--regularization', action='store',
                         type='int', dest='regularize', default=0,
                         help='L2 regularization in LSTM')
    optParser.add_option('-l', '--bilstm', action='store',
                         type='int', dest='bilstm', default=0,
                         help='Use bilstm')
    optParser.add_option('-p', '--postags', action='store',
                         type='int', dest='postags', default=0,
                         help='Include postags in the model')
    optParser.add_option('-s', '--srltags', action='store',
                         type='int', dest='srltags', default=0,
                         help='Include srltags in the model')
    optParser.add_option('-e', '--epochs', action='store',
                         type='int', dest='epochs', default=2,
                         help='Number of epochs')
    optParser.add_option('-d', '--data', action='store',
                         type='string', dest='datadir', default='data',
                         help='Base data dir')
    optParser.add_option('-o', '--output', action='store',
                         type='string', dest='outputdir', default='.',
                         help='Base output dir')
    optParser.add_option('-u', '--chunk', action='store',
                         type='int', dest='chunk', default='0',
                         help='Include chunk tags in the model')
    optParser.add_option('-v', '--verbs', action='store',
                         type='int', dest='verbs', default='0',
                         help='Include verbs in the model')
    optParser.add_option('-w', '--commonwords', action='store',
                         type='int', dest='commonwords', default='0',
                         help='Include common words in the model')
    optParser.add_option('-m', '--siamese', action='store',
                         type='int', dest='siamese', default='0',
                         help='Siamese architecture')
    optParser.add_option('-k', '--cwnolstm', action='store',
                         type='int', dest='cwnolstm', default='0',
                         help='Not use LSTM for Common words')

    opts, args = optParser.parse_args()
    return opts


# Main execution

opts = parseOptions()
print get_current_time(), 'Command line options: ', opts
NUM_EPOCHS = opts.epochs
if opts.baseline == 0:
    print get_current_time(), 'Running latest version.'
else:
    print get_current_time(), 'Running baseline version.'

output_dir = opts.outputdir
print get_current_time(), 'Starting execution . . .'
print get_current_time(), 'Setting output directory to', output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

pos_tags = {}
srl_tags = {}
chunk_tags = {}

data_dir = opts.datadir
print get_current_time(), 'Setting data directory to', data_dir

data = pd.read_csv(data_dir + '/quora_duplicate_questions.tsv', sep='\t')
features = pd.read_csv(data_dir + '/quora_features.csv')
additional_features = pd.read_csv(data_dir + '/quora_additional_features.csv')

data_len = len(data)
train_data_len = (int)(math.floor(data_len * 0.9))
test_data_len = (int)(data_len - train_data_len)

train_data = data[0:train_data_len]
test_data = data[train_data_len + 1:]

features_train = features[0:train_data_len]
additional_features_train = additional_features[0:train_data_len]

features_test = features[train_data_len + 1:]
additional_features_test = additional_features[train_data_len + 1:]

common_words_train = features_train.common_words.values
common_words_test = features_test.common_words.values
common_words_dim = max(max(common_words_train), max(common_words_test)) + 1

print get_current_time(), 'type(common_words): ', type(common_words_train)
print get_current_time(), 'common_words.shape: ', common_words_train.shape

train_y = train_data.is_duplicate.values
test_y = test_data.is_duplicate.values

tk = text.Tokenizer(nb_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(train_data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

print get_current_time(), 'type(x1): ', type(x1)
print get_current_time(), 'x1.shape: ', x1.shape

x2 = tk.texts_to_sequences(train_data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

x1_test = tk.texts_to_sequences(test_data.question1.values)
x1_test = sequence.pad_sequences(x1_test, maxlen=max_len)
x2_test = tk.texts_to_sequences(test_data.question2.values.astype(str))
x2_test = sequence.pad_sequences(x2_test, maxlen=max_len)

if opts.postags == 1:
    print get_current_time(), 'Getting POS tags for q1 set . . .'
    q1_pos_tags = get_pos_tag_sequence_padded(additional_features_train.pos_tags1, max_len)
    q1_pos_tags_test = get_pos_tag_sequence_padded(additional_features_test.pos_tags1, max_len)
    print get_current_time(), 'q1_pos_tags: ', q1_pos_tags
    print get_current_time(), 'q1_pos_tags_test: ', q1_pos_tags_test

    print get_current_time(), 'Getting POS tags for q2 set . . .'
    q2_pos_tags = get_pos_tag_sequence_padded(additional_features_train.pos_tags2.values, max_len)
    q2_pos_tags_test = get_pos_tag_sequence_padded(additional_features_test.pos_tags2.values, max_len)
    print get_current_time(), 'q2_pos_tags: ', q2_pos_tags
    print get_current_time(), 'q2_pos_tags_test: ', q2_pos_tags_test

srl_max_len = 60
if opts.srltags == 1:
    print get_current_time(), 'Getting SRL tags for q1 set . . .'
    q1_srl_tags = get_srl_tag_sequence_padded(additional_features_train.srl1.values, srl_max_len)
    q1_srl_tags_test = get_srl_tag_sequence_padded(additional_features_test.srl1.values, srl_max_len)
    print get_current_time(), 'q1_srl_tags: ', q1_srl_tags
    print get_current_time(), 'q1_srl_tags_test: ', q1_srl_tags_test

    print get_current_time(), 'Getting SRL tags for q2 set . . .'
    q2_srl_tags = get_srl_tag_sequence_padded(additional_features_train.srl2.values, srl_max_len)
    q2_srl_tags_test = get_srl_tag_sequence_padded(additional_features_test.srl2.values, srl_max_len)
    print get_current_time(), 'q2_srl_tags: ', q2_srl_tags
    print get_current_time(), 'q2_srl_tags_test: ', q2_srl_tags_test

if opts.chunk == 1:
    print get_current_time(), 'Getting chunk tags for q1 set . . .'
    q1_chunk_tags = get_chunk_tag_sequence_padded(additional_features_train.chunk1, max_len)
    q1_chunk_tags_test = get_chunk_tag_sequence_padded(additional_features_test.chunk1, max_len)
    print get_current_time(), 'q1_chunk_tags: ', q1_chunk_tags
    print get_current_time(), 'q1_chunk_tags_test: ', q1_chunk_tags_test

    print get_current_time(), 'Getting chunk tags for q2 set . . .'
    q2_chunk_tags = get_chunk_tag_sequence_padded(additional_features_train.chunk2, max_len)
    q2_chunk_tags_test = get_chunk_tag_sequence_padded(additional_features_test.chunk2, max_len)
    print get_current_time(), 'q2_chunk_tags: ', q2_chunk_tags
    print get_current_time(), 'q2_chunk_tags_test: ', q2_chunk_tags_test

word_index = tk.word_index

y_train_enc = np_utils.to_categorical(train_y)

print get_current_time(), 'Generating embedding index . . .'
embeddings_index = {}
f = open(data_dir + '/glove.840B.300d.txt')
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

models = []
model_inputs = []
model_inputs_test = []

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
models.append(model1)
model_inputs.append(x1)
model_inputs_test.append(x1_test)

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
models.append(model2)
model_inputs.append(x2)
model_inputs_test.append(x2_test)

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
models.append(model3)
model_inputs.append(x1)
model_inputs_test.append(x1_test)

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
models.append(model4)
model_inputs.append(x2)
model_inputs_test.append(x2_test)

# model5
if opts.siamese == 0:
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
    models.append(model5)
    model_inputs.append(x1)
    model_inputs_test.append(x1_test)

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
    models.append(model6)
    model_inputs.append(x2)
    model_inputs_test.append(x2_test)
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

    input_a = Input(shape=(40,))
    input_b = Input(shape=(40,))

    processed_a = model5(input_a)
    processed_b = model5(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model5s = Model(input=[input_a, input_b], output=distance)
    models.append(model5s)
    model_inputs.append(x1)
    model_inputs.append(x2)

    model_inputs_test.append(x1_test)
    model_inputs_test.append(x2_test)

if opts.commonwords == 1:
    model7 = Sequential()
    if opts.cwnolstm == 1:
        model7.add(Embedding(common_words_dim, 300, input_length=1))
        model7.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
    else:
        model7.add(Embedding(common_words_dim, 10, input_length=1))
        model7.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))
    models.append(model7)
    model_inputs.append(common_words_train)
    model_inputs_test.append(common_words_test)

if opts.postags == 1:
    model8 = Sequential()
    model8.add(Embedding(len(pos_tags) + 1, 10, input_length=max_len))
    model8.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    model9 = Sequential()
    model9.add(Embedding(len(pos_tags) + 1, 10, input_length=max_len))
    model9.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    models.append(model8)
    models.append(model9)

    model_inputs.append(q1_pos_tags)
    model_inputs.append(q2_pos_tags)

    model_inputs_test.append(q1_pos_tags_test)
    model_inputs_test.append(q2_pos_tags_test)

if opts.srltags == 1:
    model10 = Sequential()
    model10.add(Embedding(len(srl_tags) + 1, 10, input_length=srl_max_len))
    model10.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    model11 = Sequential()
    model11.add(Embedding(len(srl_tags) + 1, 10, input_length=srl_max_len))
    model11.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    models.append(model10)
    models.append(model11)

    model_inputs.append(q1_srl_tags)
    model_inputs.append(q2_srl_tags)

    model_inputs_test.append(q1_srl_tags_test)
    model_inputs_test.append(q2_srl_tags_test)

if opts.chunk == 1:
    model12 = Sequential()
    model12.add(Embedding(len(chunk_tags) + 1, 10, input_length=max_len))
    model12.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    model13 = Sequential()
    model13.add(Embedding(len(chunk_tags) + 1, 10, input_length=max_len))
    model13.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))

    models.append(model12)
    models.append(model13)

    model_inputs.append(q1_chunk_tags)
    model_inputs.append(q2_chunk_tags)

    model_inputs_test.append(q1_chunk_tags_test)
    model_inputs_test.append(q2_chunk_tags_test)

merged_model = Sequential()
if opts.baseline == 1:
    if opts.siamese == 1:
        merged_model.add(Merge([model1, model2, model3, model4, model5s], mode='concat'))
    else:
        merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode='concat'))
else:
    merged_model.add(Merge(models, mode='concat'))

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

checkpoint = ModelCheckpoint(output_dir + '/weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

result = []
if opts.baseline == 1:
    result = merged_model.fit([x1, x2, x1, x2, x1, x2], y=train_y, batch_size=384, nb_epoch=NUM_EPOCHS,
                              verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
else:
    result = merged_model.fit(model_inputs, y=train_y, batch_size=384, nb_epoch=NUM_EPOCHS,
                              verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])

print get_current_time(), 'Result keys: ', result.history.keys()
print get_current_time(), 'Training accuracy: ', result.history['acc']
print get_current_time(), 'Validation accuracy: ', result.history['val_acc']
file_suffix = '_siamese_' + str(opts.siamese) + '_attention_' + str(opts.attention) + '_srl_' + str(opts.srltags) + '_pos_' + str(opts.postags) + '_bilstm_' \
              + str(opts.bilstm) + '_cnn_' + str(opts.cnn) + '_epochs_' + str(opts.epochs) + '_regularize_' + str(opts.regularize) \
              + '_chunk_' + str(opts.chunk) + '.png'


#plot Accuracy
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_dir + '/accuracy' + file_suffix)
plt.clf()

#Plot loss
print get_current_time(), 'Training loss: ', result.history['loss']
print get_current_time(), 'Validation loss: ', result.history['val_loss']
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_dir + '/loss' + file_suffix)
plt.clf()

print get_current_time(), 'Testing the model on test set . . .'
if opts.baseline == 1:
    score, acc = merged_model.evaluate([x1_test, x2_test, x1_test, x2_test, x1_test, x2_test], test_y, batch_size=384)
else:
    score, acc = merged_model.evaluate(model_inputs_test, test_y, batch_size=384)
print('Test score:', score)
print('Test accuracy:', acc)

"""
Detecting duplicate quora questions
additional feature engineering like POS tags, SRL tags, verbs etc
@author: Pandian Raju
"""

import pandas as pd
import nltk as nt
import sys
from practnlptools.tools import Annotator
import time
import datetime


def get_current_time():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + ' '


def get_pos_tags(question):
    try:
        tokens = nt.word_tokenize(question)
        tags = nt.pos_tag(tokens)
        return tags
    except:
        return []


def get_annotations(question):
    annotator = Annotator()
    annotations = annotator.getAnnotations(question)
    srl = annotations['srl']
    verbs = annotations['verbs']
    ner = annotations['ner']
    chunk = annotations['chunk']
    return srl, verbs, ner, chunk

input_file = sys.argv[1]
output_file = sys.argv[2]

print get_current_time(), 'Input file: ', input_file
print get_current_time(), 'Output file: ', output_file

data = pd.read_csv(input_file, sep='\t')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

print get_current_time(), 'Getting POS tags for question set 1 . . .'
data['pos_tags1'] = data.question1.apply(lambda x: get_pos_tags(x))
print get_current_time(), 'Getting POS tags for question set 2 . . .'
data['pos_tags2'] = data.question2.apply(lambda x: get_pos_tags(x))

print get_current_time(), 'Getting practnlptools annotations for the question set 1 . . .'
srl = []
ner = []
verbs = []
chunk = []
i = 1
for q1 in data.question1.values:
    try:
        srl1, ner1, verbs1, chunk1 = get_annotations(q1)
        srl.append(srl1)
        ner.append(ner1)
        verbs.append(verbs1)
        chunk.append(chunk1)
    except:
        print 'Exception. Appending empty annotations..'
        srl.append([])
        ner.append([])
        verbs.append([])
        chunk.append([])
    print get_current_time(), 'Processed ', i, ' sentences. '
    i += 1
data['srl1'] = srl
data['ner1'] = ner
data['verbs1'] = verbs
data['chunk1'] = chunk

print get_current_time(), 'Getting practnlptools annotations for the question set 2 . . .'
srl = []
ner = []
verbs = []
chunk = []
i = 1
for q2 in data.question2.values:
    try:
        srl2, ner2, verbs2, chunk2 = get_annotations(q2)
        srl.append(srl2)
        ner.append(ner2)
        verbs.append(verbs2)
        chunk.append(chunk2)
    except:
        print 'Exception. Appending empty annotations..'
        srl.append([])
        ner.append([])
        verbs.append([])
        chunk.append([])
    print get_current_time(), 'Processed ', i, ' sentences. '
    i += 1
data['srl2'] = srl
data['ner2'] = ner
data['verbs2'] = verbs
data['chunk2'] = chunk

data.to_csv(output_file, index=False)

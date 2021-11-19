import warnings
warnings.filterwarnings('ignore')

import tqdm as tqdm
import shutil

import time
from contextlib import contextmanager
import os, argparse
import spacy, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import gc

import csv
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from sklearn.metrics import classification_report

import math
import numpy as np
import pickle
import operator
from operator import itemgetter
from itertools import zip_longest
from collections import defaultdict
import json
import joblib
from tqdm import tqdm
import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
import datetime

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU, add, Conv2D, Reshape
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, multiply
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import Sequence
from tensorflow.keras import utils
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image, text, sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

import silence_tensorflow.auto
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score

import wandb

wandb.init(project="it_project")

def getfreq(answers_str):
	answers = answers_str.split(';')
	return max(set(answers), key = answers.count)

def process_question_annotation(subset):
    if subset == 'train2014':
      with open('/home3/181ee103/it_project/attention/ta/cleaned_answers_v2_ta.txt', 'r', encoding = "utf8") as f:
        anno = f.readlines()
      with open('/home3/181ee103/it_project/attention/ta/cleaned_questions_v2_ta.txt', 'r', encoding = "utf8") as f:
        ques = f.readlines()
      with open('/home3/181ee103/it_project/attention/ta/cleaned_image_id_v2_ta.txt', 'r', encoding = "utf8") as f:
        img_ids = f.readlines()
    else:\validation\ta
      with open('/home3/181ee103/it_project/validation/ta/val_cleaned_answers_v2_ta.txt', 'r', encoding = "utf8") as f:
        anno = f.readlines()
      with open('/home3/181ee103/it_project/validation/ta/val_cleaned_questions_v2_ta.txt', 'r', encoding = "utf8") as f:
        ques = f.readlines()
      with open('/home3/181ee103/it_project/validation/ta/val_cleaned_image_id_v2_ta.txt', 'r', encoding = "utf8") as f:
        img_ids = f.readlines()
    
    imdir='%s/COCO_%s_%012d.jpg' ## COCO_train2014_000000291417.jpg
    data = []

    for i in tqdm(range(len(anno))):
        image_path = imdir%(subset, subset, int(img_ids[i]))
        question = ques[i][:-1]
        answer = getfreq(anno[i])
        answer_list = anno[i][:-1]

        data.append({'img_path': image_path, 'question': question, 'answer': answer, 'answer_list': answer_list})
    
    json.dump(data, open(f'/home3/181ee103/vqa_raw_{subset}.json', 'w'))

subset = ['train2014', 'val2014']

with timer('Processing Questions nd Annotations:'):
  for x in subset:
    process_question_annotation(x)

train_data = json.load(open(f'/home3/181ee103/vqa_raw_train2014.json', 'r'))
val_data = json.load(open(f'/home3/181ee103/vqa_raw_val2014.json', 'r'))

answer_freq= defaultdict(int)
for answer in list(map(itemgetter('answer'), train_val_data)):
		answer_freq[answer] += 1
print('There are total ', len(answer_freq), ' different types of answers.')

def selectTopAnswersData(questions_list, answer_list, answers_list, images_list, k):
	answer_freq= defaultdict(int)

	for answer in answer_list:
		answer_freq[answer] += 1

	sorted_freq = sorted(answer_freq.items(), key=operator.itemgetter(1), reverse=True)[0: k]
	top_answers, top_freq = zip(*sorted_freq)
 
	new_questions_list=[]
	new_answer_list=[]
	new_answers_list=[]
	new_images_list=[]

	for question, answer, answers, image in zip(questions_list, answer_list, answers_list, images_list):
		if answer in top_answers:
			new_questions_list.append(question)
			new_answer_list.append(answer)
			new_answers_list.append(answers)
			new_images_list.append(image)
	
	print('Data size reduced by', np.round(((len(questions_list)-len(new_questions_list))/len(questions_list))*100, 2),'%')
	return(new_questions_list, new_answer_list, new_answers_list, new_images_list, top_answers)

max_answers = 1000
max_seq_len = 33
EPOCHS      = 500

dim_d       = 512
dim_k       = 256
l_rate      = 1e-4
d_rate      = 0.5
reg_value   = 0.01

questions_train, answer_train, answers_train, images_train, top_answers = selectTopAnswersData(list(map(itemgetter('question'), train_data)), 
                                                                                               list(map(itemgetter('answer'), train_data)), 
                                                                                               list(map(itemgetter('answer_list'), train_data)), 
                                                                                               list(map(itemgetter('img_path'), train_data)), max_answers)

def filterTopAnswersData(questions_list, answer_list, answers_list, images_list, top_answers):
	new_questions_list=[]
	new_answer_list=[]
	new_answers_list=[]
	new_images_list=[]

	for question, answer, answers, image in zip(questions_list, answer_list, answers_list, images_list):
		if answer in top_answers:
			new_questions_list.append(question)
			new_answer_list.append(answer)
			new_answers_list.append(answers)
			new_images_list.append(image)
	
	print('Data size reduced by', np.round(((len(questions_list)-len(new_questions_list))/len(questions_list))*100, 2),'%')
	return (new_questions_list, new_answer_list, new_answers_list, new_images_list)

questions_val, answer_val, answers_val, images_val = filterTopAnswersData(list(map(itemgetter('question'), val_data)), 
                                                                          list(map(itemgetter('answer'), val_data)),
                                                                          list(map(itemgetter('answer_list'), val_data)), 
                                                                          list(map(itemgetter('img_path'), val_data)), top_answers)

with open('/home3/181ee103/vqa_raw_train2014_top1000.json', 'wb') as f:
  joblib.dump((questions_train, answer_train, answers_train, images_train), f)

with open('/home3/181ee103/vqa_raw_val2014_top1000.json', 'wb') as f:
  joblib.dump((questions_val, answer_val, answers_val, images_val), f)

with open('/home3/181ee103/vqa_raw_train2014_top1000.json', 'rb') as f:
  questions_train, answer_train, answers_train, images_train = joblib.load(f)

with open('/home3/181ee103/vqa_raw_val2014_top1000.json', 'rb') as f:
  questions_val, answer_val, answers_val, images_val = joblib.load(f)

image_folder = '/train2014/'

def image_feature_extractor(target_path, image_list, BATCH_SIZE):
	model = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(3, 224, 224)))
	progbar = utils.Progbar(int(np.ceil(len(image_list) / float(BATCH_SIZE))))

	for (b, i) in enumerate(range(0, len(image_list), BATCH_SIZE)):
		progbar.update(b+1)
		
		batch_range = range(i, min(i + BATCH_SIZE, len(image_list)))
		batchPaths = image_list[batch_range[0]: batch_range[-1]+1]

		batchImages = []
		batchIds = []
		for imagePath in batchPaths:
			img = image.load_img('/home3/181ee103/train2014/'+imagePath, target_size=(224, 224))
			img = image.img_to_array(img)
			img = np.expand_dims(img, axis=0)
			img = preprocess_input(img)
			batchImages.append(img)
			batchIds.append(imagePath.split('.')[0][-6:])
	  
		batchImages = np.vstack(batchImages) # (BATCH_SIZE, 3, 224, 224)
		features = model.predict(batchImages) # (BATCH_SIZE, 512, 7, 7)
		features = tf.reshape(features, (features.shape[0], features.shape[1], -1)) # (BATCH_SIZE, 512, 49)
		features = tf.transpose(features, perm =[0,2,1])  # (BATCH_SIZE, 49, 512)
		for id, feat in zip(batchIds, features):
			np.save(os.path.join(target_path, id), feat)

image_list = os.listdir('train2014')
BATCH_SIZE = 300
target_path = '/home3/181ee103/features'

#image_feature_extractor(target_path, image_list, BATCH_SIZE)

import re

def process_sentence(sentence):
    periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip   = re.compile("(\d)(\,)(\d)")
    punct        = [';', r"/", '[', ']', '"', '{', '}',
                    '(', ')', '=', '+', '\\', '_', '-',
                    '>', '<', '@', '`', ',', '?', '!']

    inText = sentence.replace('\n', ' ')
    inText = inText.replace('\t', ' ')
    inText = inText.strip()
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    outText = ' '.join(outText)
    return outText

questions_train_processed = pd.Series(questions_train).apply(process_sentence)
questions_val_processed = pd.Series(questions_val).apply(process_sentence)

with open('/home3/181ee103/att_ta_processed_questions.pkl', 'wb') as f:
  joblib.dump((questions_train_processed, questions_val_processed), f)

with open('/home3/181ee103/att_ta_processed_questions.pkl', 'rb') as f:
  questions_train_processed, questions_val_processed = joblib.load(f)

tok=text.Tokenizer(filters='')
tok.fit_on_texts(questions_train_processed)

with open('/home3/181ee103/att_ta_text_tokenizer.pkl', 'wb') as f:
  joblib.dump(tok, f)

with open('/home3/181ee103/att_ta_text_tokenizer.pkl', 'rb') as f:
  tok = joblib.load(f)

question_data_train = tok.texts_to_sequences(questions_train_processed)
question_data_val = tok.texts_to_sequences(questions_val_processed)

MAX_LEN = 33
vocab_size  = len(tok.word_index) + 1

question_data_train=sequence.pad_sequences(question_data_train, maxlen=MAX_LEN, padding='post')
question_data_val=sequence.pad_sequences(question_data_val, maxlen=MAX_LEN, padding = 'post')

with open('/home3/181ee103/att_ta_tokenised_data_post.pkl', mode='wb') as f:
  pickle.dump((question_data_train, question_data_val), f)

with open('/home3/181ee103/att_ta_tokenised_data_post.pkl', mode='rb') as f:
  question_data_train, question_data_val = pickle.load(f)

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answer_train)

with open('/home3/181ee103/att_ta_labelencoder.pkl', 'wb') as f:
  joblib.dump(labelencoder, f)

with open('/home3/181ee103/att_ta_labelencoder.pkl', 'rb') as f:
  labelencoder = joblib.load(f)

def get_answers_matrix(answers, encoder):
	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = utils.to_categorical(y, nb_classes)
	return Y

sss = StratifiedShuffleSplit(n_splits=1, test_size= 0.25, random_state=42)

for train_index, val_index in sss.split(images_train, answer_train):
  TRAIN_INDEX = train_index
  VAL_INDEX = val_index

image_list_tr, image_list_vl = np.array(images_train)[TRAIN_INDEX.astype(int)], np.array(images_train)[VAL_INDEX.astype(int)]
question_tr, question_vl = question_data_train[TRAIN_INDEX], question_data_train[VAL_INDEX]
answer_matrix = get_answers_matrix(answer_train, labelencoder)
answer_tr, answer_vl = answer_matrix[TRAIN_INDEX], answer_matrix[VAL_INDEX]

class AttentionMaps(tf.keras.layers.Layer):
  def __init__(self, dim_k, reg_value, **kwargs):
    super(AttentionMaps, self).__init__(**kwargs)

    self.dim_k = dim_k
    self.reg_value = reg_value

    self.Wv = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=2))
    self.Wq = Dense(self.dim_k, activation=None,\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=3))

  def call(self, image_feat, ques_feat):
    C = tf.matmul(ques_feat, tf.transpose(image_feat, perm=[0,2,1])) # [b, 23, 49]
    C = tf.keras.activations.tanh(C) 
    WvV = self.Wv(image_feat)                             # [b, 49, dim_k]
    WqQ = self.Wq(ques_feat)                              # [b, 23, dim_k]
    WqQ_C = tf.matmul(tf.transpose(WqQ, perm=[0,2,1]), C) # [b, k, 49]
    WqQ_C = tf.transpose(WqQ_C, perm =[0,2,1])            # [b, 49, k]
    WvV_C = tf.matmul(tf.transpose(WvV, perm=[0,2,1]), tf.transpose(C, perm=[0,2,1]))                          
    WvV_C = tf.transpose(WvV_C, perm =[0,2,1])            # [b, 23, k]
    H_v = WvV + WqQ_C                                     # (Wv)V + ((Wq)Q)C
    H_v = tf.keras.activations.tanh(H_v)                  # tanh((Wv)V + ((Wq)Q)C) 
    H_q = WqQ + WvV_C                                     # (Wq)Q + ((Wv)V)CT
    H_q = tf.keras.activations.tanh(H_q)                  # tanh((Wq)Q + ((Wv)V)CT) 
        
    return [H_v, H_q]                                     # [b, 49, k], [b, 23, k]
  
  def get_config(self):
    config = {
        'dim_k': self.dim_k,
        'reg_value': self.reg_value
    }
    base_config = super(AttentionMaps, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

layer = AttentionMaps(64, 0.001)
config = layer.get_config()
print(config)
new_layer = AttentionMaps.from_config(config)

class ContextVector(tf.keras.layers.Layer):
  def __init__(self, reg_value, **kwargs):
    super(ContextVector, self).__init__(**kwargs)

    self.reg_value = reg_value

    self.w_hv = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=4))
    self.w_hq = Dense(1, activation='softmax',\
                        kernel_regularizer=tf.keras.regularizers.l2(self.reg_value),\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=5)) 
    
  def call(self, image_feat, ques_feat, H_v, H_q):
    a_v = self.w_hv(H_v)                               # [b, 49, 1]
    a_q = self.w_hq(H_q)                               # [b, 23, 1]
    v = a_v * image_feat                               # [b, 49, dim_d]
    v = tf.reduce_sum(v, 1)                            # [b, dim_d]
    q = a_q * ques_feat                                # [b, 23, dim_d]
    q = tf.reduce_sum(q, 1)                            # [b, dim_d]
    return [v, q]

  def get_config(self):
    config = {
        'reg_value': self.reg_value
    }
    base_config = super(ContextVector, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

layer = ContextVector(0.001)
config = layer.get_config()
print(config)
new_layer = ContextVector.from_config(config)

class PhraseLevelFeatures(tf.keras.layers.Layer):
  def __init__(self, dim_d, **kwargs):
    super(PhraseLevelFeatures, self).__init__(**kwargs)
    
    self.dim_d = dim_d
    
    self.conv_unigram = Conv1D(self.dim_d, kernel_size=1, strides=1,\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=6)) 
    self.conv_bigram =  Conv1D(self.dim_d, kernel_size=2, strides=1, padding='same',\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=7)) 
    self.conv_trigram = Conv1D(self.dim_d, kernel_size=3, strides=1, padding='same',\
                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=8)) 


  def call(self, word_feat):
    x_uni = self.conv_unigram(word_feat)                    # [b, 23, dim_d]
    x_bi  = self.conv_bigram(word_feat)                     # [b, 23, dim_d]
    x_tri = self.conv_trigram(word_feat)                    # [b, 23, dim_d]
    x = tf.concat([tf.expand_dims(x_uni, -1),\
                    tf.expand_dims(x_bi, -1),\
                    tf.expand_dims(x_tri, -1)], -1)         # [b, 23, dim_d, 3]
    x = tf.reduce_max(x, -1)                                # [b, 23, dim_d]
    return x

  def get_config(self):
    config = {
        'dim_d': self.dim_d
    }
    base_config = super(PhraseLevelFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

layer = PhraseLevelFeatures(32)
config = layer.get_config()
print(config)
new_layer = PhraseLevelFeatures.from_config(config)

def build_model(max_answers, max_seq_len, vocab_size, dim_d, dim_k, l_rate, d_rate, reg_value):
    image_input = Input(shape=(49, 512, ), name='Image_Input')
    ques_input = Input(shape=(33, ), name='Question_Input')

    image_feat = Dense(dim_d, activation=None, name='Image_Feat_Dense',\
                            kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=1))(image_input)
    image_feat = Dropout(d_rate, seed=1)(image_feat)

    ques_feat_w = Embedding(input_dim=vocab_size, output_dim=dim_d, input_length=max_seq_len,\
                            mask_zero=True)(ques_input)
    
    Hv_w, Hq_w = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Word')(image_feat, ques_feat_w)
    v_w, q_w = ContextVector(reg_value, name='ContextVector_Word')(image_feat, ques_feat_w, Hv_w, Hq_w)
    feat_w = tf.add(v_w,q_w)
    h_w = Dense(dim_d, activation='tanh', name='h_w_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=13))(feat_w)

    ques_feat_p = PhraseLevelFeatures(dim_d, name='PhraseLevelFeatures')(ques_feat_w)

    Hv_p, Hq_p = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Phrase')(image_feat, ques_feat_p)
    v_p, q_p = ContextVector(reg_value, name='ContextVector_Phrase')(image_feat, ques_feat_p, Hv_p, Hq_p)
    feat_p = concatenate([tf.add(v_p,q_p), h_w], -1) 
    h_p = Dense(dim_d, activation='tanh', name='h_p_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=14))(feat_p)

    ques_feat_s = LSTM(dim_d, return_sequences=True, input_shape=(None, max_seq_len, dim_d),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=16))(ques_feat_p)

    Hv_s, Hq_s = AttentionMaps(dim_k, reg_value, name='AttentionMaps_Sent')(image_feat, ques_feat_s)
    v_s, q_s = ContextVector(reg_value, name='ContextVector_Sent')(image_feat, ques_feat_p, Hv_s, Hq_s)
    feat_s = concatenate([tf.add(v_s,q_s), h_p], -1) 
    h_s = Dense(2*dim_d, activation='tanh', name='h_s_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=15))(feat_s)

    z   = Dense(2*dim_d, activation='tanh', name='z_Dense',\
                    kernel_regularizer=tf.keras.regularizers.l2(reg_value),\
                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=16))(h_s)
    z   = Dropout(d_rate, seed=16)(z)

    result = Dense(max_answers, activation='softmax')(z)

    model = Model(inputs=[image_input, ques_input], outputs=result)

    return model

BATCH_SIZE = 300
BUFFER_SIZE = 5000

def map_func(img_name, ques, ans):
    img_tensor = np.load('/home3/181ee103/features/' + img_name.decode('utf-8').split('.')[0][-6:] + '.npy')
    return img_tensor, ques, ans

dataset_tr = tf.data.Dataset.from_tensor_slices((image_list_tr, question_tr, answer_tr))

dataset_tr = dataset_tr.map(lambda item1, item2, item3: tf.numpy_function(
    map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_tr = dataset_tr.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_tr = dataset_tr.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset_vl = tf.data.Dataset.from_tensor_slices((image_list_vl, question_vl, answer_vl))

dataset_vl = dataset_vl.map(lambda item1, item2, item3: tf.numpy_function(
    map_func, [item1, item2, item3], [tf.float32, tf.int32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_vl = dataset_vl.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset_vl = dataset_vl.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

base_path = '/home3/181ee103/att_ta_temps'
tf.keras.backend.set_image_data_format("channels_last")

model = build_model(max_answers, max_seq_len, vocab_size, dim_d, dim_k, l_rate, d_rate, reg_value)

steps_per_epoch = int(np.ceil(len(image_list_tr)/BATCH_SIZE))
boundaries      = [50*steps_per_epoch]
values          = [l_rate, l_rate/10]

learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer        = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

loss_object      = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='auto')

checkpoint_directory = base_path+"/training_checkpoints/"+str(l_rate)+"_"+str(dim_k)
SAVE_CKPT_FREQ = 5

ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=3)

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

train_score = F1Score(num_classes=max_answers, average='micro', name='train_score')
val_score = F1Score(num_classes=max_answers, average='micro', name='val_score')

train_log_dir = base_path+'/logs/'+str(l_rate)+"_"+str(dim_k)+'/train'
val_log_dir   = base_path+'/logs/'+str(l_rate)+"_"+str(dim_k)+'/validation'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

def train_step(model, img, ques, ans, optimizer):
  with tf.GradientTape() as tape:
    predictions = model([img, ques], training=True)
    loss = loss_object(ans, predictions)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_score(ans, predictions)

  grads_ = list(zip(grads, model.trainable_variables))
  return grads_

def test_step(model, img, ques, ans):
  predictions = model([img, ques])
  loss = loss_object(ans, predictions)

  val_loss(loss)
  val_score(ans, predictions)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print("Restored from {}".format(manager.latest_checkpoint))
    START_EPOCH = int(manager.latest_checkpoint.split('-')[-1]) * SAVE_CKPT_FREQ
    print("Resume training from epoch: {}".format(START_EPOCH))
else:
    print("Initializing from scratch")
    START_EPOCH = 0

for epoch in range(START_EPOCH, EPOCHS):

  start = time.time()

  for img, ques, ans in (dataset_tr):
    grads = train_step(model, img, ques, ans, optimizer)

  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('f1_score', train_score.result(), step=epoch)
    for var in model.trainable_variables:
        tf.summary.histogram(var.name, var, step=epoch)
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad, step=epoch)

  for img, ques, ans in (dataset_vl):
    test_step(model, img, ques, ans)
  
  with val_summary_writer.as_default():
    tf.summary.scalar('loss', val_loss.result(), step=epoch)
    tf.summary.scalar('f1_score', val_score.result(), step=epoch)
  
  template = 'Epoch {}, loss: {:.4f}, f1_score: {:.4f}, val loss: {:.4f}, val f1_score: {:.4f}, time: {:.0f} sec'
  wandb.log({"Epoch": epoch + 1})
  wandb.log({"Train Loss": train_loss.result()})
  wandb.log({"Train F1 Score": train_score.result()})
  wandb.log({"Validation Loss": val_loss.result()})
  wandb.log({"Validation F1 Score": val_score.result()})

  print (template.format(epoch + 1,
                         train_loss.result(), 
                         train_score.result(),
                         val_loss.result(), 
                         val_score.result(),
                         (time.time() - start)))
  train_loss.reset_states()
  train_score.reset_states()
  val_loss.reset_states()
  val_score.reset_states()

  ckpt.step.assign_add(1)
  if int(ckpt.step) % SAVE_CKPT_FREQ == 0:
      manager.save()
      print('Saved checkpoint.')

tr_questions = np.array(questions_train_processed)[TRAIN_INDEX.astype(int)]
val_questions = np.array(questions_train_processed)[VAL_INDEX.astype(int)]

tr_answers = np.array(answers_train)[TRAIN_INDEX.astype(int)]
val_answers = np.array(answers_train)[VAL_INDEX.astype(int)]

tr_answer = np.array(answer_train)[TRAIN_INDEX.astype(int)]
val_answer = np.array(answer_train)[VAL_INDEX.astype(int)]

tr_images = np.array(images_train)[TRAIN_INDEX.astype(int)]
val_images = np.array(images_train)[VAL_INDEX.astype(int)]

def map_func_eval(img_name, ques):
    img_tensor = np.load('/home3/181ee103/features/' + img_name.decode('utf-8').split('.')[0][-6:] + '.npy')
    return img_tensor, ques

dataset_tr_eval = tf.data.Dataset.from_tensor_slices((image_list_tr, question_tr))

dataset_tr_eval = dataset_tr_eval.map(lambda item1, item2: tf.numpy_function(
    map_func_eval, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_tr_eval = dataset_tr_eval.batch(BATCH_SIZE)
dataset_vl_eval = tf.data.Dataset.from_tensor_slices((image_list_vl, question_vl))

dataset_vl_eval = dataset_vl_eval.map(lambda item1, item2: tf.numpy_function(
    map_func_eval, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset_vl_eval = dataset_vl_eval.batch(BATCH_SIZE)

def predict_answers(dataset, model, labelencoder):
    predictions = []
    for img, ques in (dataset):
        preds = model([img, ques])
        predictions.extend(preds)

    y_classes = tf.argmax(predictions, axis=1, output_type=tf.int32)
    y_predict = (labelencoder.inverse_transform(y_classes))
    return y_predict

y_predict_text_tr = predict_answers(dataset_tr_eval, model, labelencoder)

y_predict_text_vl = predict_answers(dataset_vl_eval, model, labelencoder)

def model_metric(predictions, truths):
    total = 0
    correct_val=0.0

    for prediction, truth in zip(predictions, truths):
        
        temp_count=0
        total +=1

        for _truth in truth.split(';'):
            if prediction == _truth:
                temp_count+=1
        
        if temp_count>2:
            correct_val+=1
        else:
            correct_val+=float(temp_count)/3

    return (correct_val/total)*100

tr_score = model_metric(y_predict_text_tr, tr_answers)

print('Final Accuracy on the train set is', tr_score)
wandb.log({"Train Accuracy": tr_score})

val_score = model_metric(y_predict_text_vl, val_answers)

print('Final Accuracy on the validation set is', val_score)
wandb.log({"Validation Accuracy": val_score})
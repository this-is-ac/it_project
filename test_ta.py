import numpy as np
import warnings
warnings.filterwarnings("ignore")

import scipy.io
from sklearn.preprocessing import LabelEncoder
import pickle
import fasttext

from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras import Input
from keras.layers.recurrent import LSTM
from keras.layers import concatenate
from keras.layers.merge import Concatenate
from keras.models import model_from_json, Model
from keras.utils.vis_utils import plot_model
from collections import defaultdict
import operator
from keras.utils import np_utils, generic_utils
from itertools import zip_longest
from tqdm import tqdm
from keras.models import load_model
import matplotlib.pyplot as plt

test_questions = open('/home3/181ee103/it_project/validation/ta/val_cleaned_questions_ta.txt', 'rb').read().decode('utf-8').splitlines()
test_answers = open('/home3/181ee103/it_project/validation/ta/val_cleaned_answers_ta.txt','rb').read().decode('utf-8').splitlines()
test_image_id = open('/home3/181ee103/it_project/validation/ta/val_cleaned_image_id_ta.txt','rb').read().decode('utf-8').splitlines()
vgg_path = "/home3/181ee103/coco/vgg_feats.mat"

print("Data Successfully Loaded ")
print("Total number of Questions are {} and Answers are {}".format(len(test_questions), len(test_answers)))

nlp = fasttext.load_model('/home3/181ee103/indicnlp.ft.ta.300.bin')

vgg = scipy.io.loadmat(vgg_path)
features = vgg['feats']

batch_size               =      512
img_dim                  =     4096
word2vec_dim             =      300
num_hidden_nodes_mlp     =     1024
num_hidden_nodes_lstm    =      512
num_layers_lstm          =        5
dropout                  =       0.5
activation_mlp           =     'tanh'
num_epochs = 100

img_ids = open('/home3/181ee103/it_project/baseline/ta/coco_vgg_IDMap.txt','rb').read().decode('utf-8').splitlines()
id_map = dict()
for ids in img_ids:
    id_split = ids.split()
    id_map[id_split[0]] = int(id_split[1])
	
image_model = Sequential()
image_model.add(Reshape(input_shape = (4096,), target_shape=(4096,), name = "image"))
model1 = Model(inputs = image_model.input, outputs = image_model.output)
model1.summary()

language_model = Sequential()
language_model.add(LSTM(num_hidden_nodes_lstm, return_sequences=True, input_shape=(None, word2vec_dim), name = "sentence"))

for i in range(num_layers_lstm-2):
    language_model.add(LSTM(num_hidden_nodes_lstm, return_sequences=True))
language_model.add(LSTM(num_hidden_nodes_lstm, return_sequences=False))

model2 = Model(language_model.input, language_model.output)
model2.summary()

combined = concatenate([image_model.output, language_model.output])

model = Dense(num_hidden_nodes_mlp, activation = 'tanh')(combined)
model = Dropout(0.5)(model)

model = Dense(num_hidden_nodes_mlp, activation = 'tanh')(model)
model = Dropout(0.5)(model)

model = Dense(num_hidden_nodes_mlp, activation = 'tanh')(model)
model = Dropout(0.5)(model)

model = Dense(upper_lim)(model)
model = Activation("softmax")(model)

model = Model(inputs=[image_model.input, language_model.input], outputs=model)
model.compile(loss='categorical_crossentropy', optimizer='adam') #rmsprop
model.summary()

from indicnlp.tokenize import indic_tokenize  

def get_questions_tensor_timeseries(questions, nlp, timesteps):
    assert not isinstance(questions, list)
    nb_samples = len(questions)
    word_vec_dim = nlp.get_dimension()
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = indic_tokenize.trivial_tokenize(questions[i])
        for j in range(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = nlp.get_word_vector(tokens[j])
    return questions_tensor

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    assert not isinstance(img_coco_ids, list)
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

    return image_matrix

def get_answers_sum(answers, encoder):
    assert not isinstance(answers, list)
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y

def grouped(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
	
import tensorflow as tf
tf.config.run_functions_eagerly(True)

model = keras.models.load_model("/home3/181ee103/trained_models/ta/ta_baseline.h5")
label_encoder = pickle.load(open('/home3/181ee103/trained_models/ta/label_encoder_ta_baseline.pkl','rb'))

y_pred = []
batch_size = 512 

for qu_batch,an_batch,im_batch in zip(grouped(test_questions, batch_size, 
                                                   fillvalue=test_questions[0]), 
                                           grouped(test_answers, batch_size, 
                                                   fillvalue=test_answers[0]), 
                                           grouped(test_image_id, batch_size, 
                                                   fillvalue=test_image_id[0])):
    timesteps = len(indic_tokenize.trivial_tokenize(qu_batch[-1]))
    X_ques_batch = get_questions_tensor_timeseries(qu_batch, nlp, timesteps)
    X_img_batch = get_images_matrix(im_batch, id_map, features)
    y_predict = model.predict(({'sentence_input' : X_ques_batch, 'image_input' : X_img_batch}))
    y_predict = np.argmax(y_predict,axis=1)
    y_pred.extend(label_encoder.inverse_transform(y_predict))

pickle.dump(y_pred, open('/home3/181ee103/ta_baseline_predictions.pkl','wb'))

#with open('/home3/181ee103/ta_baseline_predictions.pkl', 'rb') as f:
#    mynewlist = pickle.load(f)	
 
correct_val = 0.0
total = 0

for pred, truth, ques, img in zip(y_pred, test_answers, test_questions, test_image_id):
    t_count = 0
    for _truth in truth.split(';'):
        if pred == truth:
            t_count += 1 
    if t_count >=1:
        correct_val +=1
    else:
        correct_val += float(t_count)/3
    total +=1
    
print ("Accuracy: ", round((correct_val/total)*100,2))

with open("/home3/181ee103/ta_baseline_acc.txt", "w") as f:
    f.write(str(round((correct_val/total)*100,5)))
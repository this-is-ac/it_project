import numpy as np
import warnings
warnings.filterwarnings("ignore")

import scipy.io
from sklearn.preprocessing import LabelEncoder
import pickle
import fasttext
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
from progressbar import Bar, ETA, Percentage, ProgressBar
import wandb

wandb.init(project="it_project")

questions = open('/home3/181ee103/it_project/baseline/ml/cleaned_questions_ml.txt', 'rb').read().decode('utf-8').splitlines()
answers = open('/home3/181ee103/it_project/baseline/ml/cleaned_answers_ml.txt','rb').read().decode('utf-8').splitlines()
image_id = open('/home3/181ee103/it_project/baseline/ml/cleaned_image_id_ml.txt','rb').read().decode('utf-8').splitlines()
vgg_path = "/home3/181ee103/coco/vgg_feats.mat"

print("Data Successfully Loaded ")
print("Total number of Questions are {} and Answers are {}".format(len(questions), len(answers)))

print("Sample Data : ")
print(questions[0])
print(answers[0])
print(image_id[0])

vgg = scipy.io.loadmat(vgg_path)
features = vgg['feats']

from transformers import AutoModel, AutoTokenizer

indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_model = AutoModel.from_pretrained("ai4bharat/indic-bert")

def freq_answers(questions, answers, image_id, upper_lim):
    freq_ans = defaultdict(int)
    for ans in answers:
        freq_ans[ans] +=1
    
    sort = sorted(freq_ans.items(), key=operator.itemgetter(1), reverse=True)[0:upper_lim]
    top_ans, top_freq = zip(*sort)
    new_answers_train = list()
    new_questions_train = list()
    new_images_train = list()
    for ans, ques, img in zip(answers, questions, image_id):
        if ans in top_ans:
            new_answers_train.append(ans)
            new_questions_train.append(ques)
            new_images_train.append(img)
    return (new_questions_train, new_answers_train, new_images_train)
	
upper_lim = 3000
questions, answers, image_id = freq_answers(questions, answers, image_id, upper_lim)
questions, answers, image_id = (list(t) for t in zip(*sorted(zip(questions, answers, image_id))))
print(len(questions), len(answers),len(image_id))

le = LabelEncoder()
le.fit(answers)
pickle.dump(le, open('/home3/181ee103/label_encoder_ml_baseline.pkl','wb'))

batch_size               =      512
img_dim                  =     4096
word2vec_dim             =      768
num_hidden_nodes_mlp     =     1024
num_hidden_nodes_lstm    =      512
num_layers_lstm          =        5
dropout                  =       0.5
activation_mlp           =     'tanh'
num_epochs = 100

img_ids = open('/home3/181ee103/it_project/baseline/ml/coco_vgg_IDMap.txt','rb').read().decode('utf-8').splitlines()
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

def get_questions_tensor_timeseries(questions, timesteps):
    assert not isinstance(questions, list)
    nb_samples = len(questions)
    word_vec_dim = 768
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = indic_tokenizer.encode(questions[i])
        model_input = indic_tokenizer(questions[i], return_tensors="pt")
        sent_output = indic_model(**model_input)
        for j in range(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = sent_output.last_hidden_state[0, j, :].detach().numpy()
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
	
from sklearn.model_selection import train_test_split

train_questions, test_questions, train_answers, test_answers, train_image_id, test_image_id = train_test_split(questions, answers, image_id, test_size = 0.2, random_state=42)

print("Total number of training samples {} and total number of test samples {} ".format(len(train_image_id), len(test_image_id)))

import tensorflow as tf
tf.config.run_functions_eagerly(True)

wandb.config = {
  "learning_rate": 0.001,
  "epochs": num_epochs,
  "batch_size": batch_size,
  "language": "ml"
}

losses = []

for k in range(num_epochs):
    print("Epoch Number: ",k+1)
    progbar = generic_utils.Progbar(len(train_questions))
    for question_batch, ans_batch, im_batch in zip(grouped(train_questions, batch_size, fillvalue=train_questions[-1]), 
                                               grouped(train_answers, batch_size, fillvalue=train_answers[-1]),
                                               grouped(train_image_id, batch_size, fillvalue=train_image_id[-1])):
        timestep = len(indic_tokenizer.encode((question_batch[-1])))
        X_ques_batch = get_questions_tensor_timeseries(question_batch, timestep)
        X_img_batch = get_images_matrix(im_batch, id_map, features)
        Y_batch = get_answers_sum(ans_batch, le)
        loss = model.train_on_batch(({'sentence_input' : X_ques_batch, 'image_input' : X_img_batch}), Y_batch)
        progbar.add(batch_size, values=[('train loss', loss)])
        losses.append(loss)
        wandb.log({"loss": loss})
    wandb.log({"Epoch":k+1})

model.save("/home3/181ee103/ml_baseline.h5")
#plt.plot(losses)
#plt.show()

label_encoder = pickle.load(open('/home3/181ee103/label_encoder_ml_baseline.pkl','rb'))

y_pred = []
batch_size = 512 

widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
pbar = ProgressBar(widgets=widgets)

for qu_batch,an_batch,im_batch in pbar(zip(grouped(test_questions, batch_size, 
                                                   fillvalue=test_questions[0]), 
                                           grouped(test_answers, batch_size, 
                                                   fillvalue=test_answers[0]), 
                                           grouped(test_image_id, batch_size, 
                                                   fillvalue=test_image_id[0]))):   
    timesteps = len(indic_tokenizer.encode((question_batch[-1])))
    X_ques_batch = get_questions_tensor_timeseries(qu_batch, timesteps)
    X_i_batch = get_images_matrix(im_batch, id_map, features)
    y_predict = model.predict(({'sentence_input' : X_ques_batch, 'image_input' : X_img_batch}))
    y_predict = np.argmax(y_predict,axis=1)
    y_pred.extend(label_encoder.inverse_transform(y_predict))

import pickle

with open('/home3/181ee103/ml_baseline_predictions.pkl', 'wb', encoding = "utf8") as f:
    pickle.dump(y_pred, f)

#with open('ml_predictions.pkl', 'rb', encoding = "utf8") as f:
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

wandb.log({"Accuracy":round((correct_val/total)*100,5)})

with open("/home3/181ee103/ml_baseline_acc.txt", "w") as f:
    f.write("%s", str(round((correct_val/total)*100,5)))
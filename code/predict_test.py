#!/usr/bin/env python
# coding: utf-8

# This kernel is continued from Save BERT fine-tuning model.
# In this kernel, you can load the weights from that kernel and make a prediction.

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'import sys
import random
import keras
import tensorflow as tf
import json
import time

sys.path.insert(0, './bert-master/')


BERT_PRETRAINED_DIR = '../model/tf_bert_model/'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
import tokenization  #Actually keras_bert contains tokenization part, here just for convenience


# ## Load raw model

# In[2]:


from keras_bert import get_model
from keras_bert  import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5,decay=0.01)
maxlen = 200
print('begin_build')
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, seq_len=maxlen)#



from keras import backend as K
class AdamWarmup(keras.optimizers.Optimizer):
    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                 amsgrad=False, **kwargs):
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
            self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_kernel_weight_decay = kernel_weight_decay
        self.initial_bias_weight_decay = bias_weight_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if 'bias' in p.name or 'Norm' in p.name:
                if self.initial_bias_weight_decay > 0.0:
                    p_t += self.bias_weight_decay * p
            else:
                if self.initial_kernel_weight_decay > 0.0:
                    p_t += self.kernel_weight_decay * p
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'decay_steps': float(K.get_value(self.decay_steps)),
            'warmup_steps': float(K.get_value(self.warmup_steps)),
            'min_lr': float(K.get_value(self.min_lr)),
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'kernel_weight_decay': float(K.get_value(self.kernel_weight_decay)),
            'bias_weight_decay': float(K.get_value(self.bias_weight_decay)),
            'amsgrad': self.amsgrad,
        }
        base_config = super(AdamWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ## Build classification model
# 
# As the Extract layer extracts only the first token where "['CLS']" used to be, we just take the layer and connect to the single neuron output.

# In[4]:


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    all_mask=[]
    longer = 0
    for i in range(example.shape[0]):
      tokens_a = tokenizer.tokenize(example[i])
      if len(tokens_a)>max_seq_length:
        tokens_a = tokens_a[:max_seq_length]
        longer += 1
        mask=[1]*(max_seq_length+2)
      else:
        mask=[1]*(len(tokens_a)+2)+[0]*(max_seq_length-len(tokens_a))
      one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
      all_tokens.append(one_token)
      all_mask.append(mask)
    return np.array(all_tokens),np.array(all_mask)
row=None
nb_epochs=1
bsz = 32
dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)


# In[5]:


print('build tokenizer done')





test_df = pd.read_csv("../data/task3/task3_new_stage2.csv")


# In[11]:


test_df['text'] = test_df['text'].astype(str)


# In[12]:


test_df['text'] = test_df['text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)


# In[13]:


test_lines = test_df['text'].values
print('sample used',test_lines.shape)
test_token_input,test_mask_input= convert_lines(test_lines,maxlen,tokenizer)
test_seg_input = np.zeros((test_token_input.shape[0],maxlen))
print(test_token_input.shape)
print(test_seg_input.shape)
print(test_mask_input.shape)
print('begin predictting')

test_x = [test_token_input,test_seg_input]


# In[14]:


from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
import keras.backend as K
import re
from keras.losses import binary_crossentropy
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import codecs

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score


#2分类的f1


class F1ScoreCallback(Callback):
    def __init__(self,validation, predict_batch_size=20, include_on_batch=False):
        super(F1ScoreCallback, self).__init__()
        self.validation = validation
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch
        
        print('validation shape',len(self.validation))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('avg_f1_score_val' in self.params['metrics']):
            self.params['metrics'].append('avg_f1_score_val')

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['avg_f1_score_val'] = float('-inf')

    def on_epoch_end(self, epoch, logs={}):
        logs['avg_f1_score_val'] = float('-inf')
            
        if (self.validation):
            y_predict = self.model.predict(self.validation[0])
            y_predict = (y_predict>0.5).astype(int)
            y_true = (self.validation[1]>0.5).astype(int)
            f1 = f1_score(y_true, y_predict)
            # print("macro f1_score %.4f " % f1)
#             f2 = f1_score(self.validation[1], y_predict, average='micro')
            # print("micro f1_score %.4f " % f2)
            avgf1=f1
            # print("avg_f1_score %.4f " % (avgf1))
            logs['avg_f1_score_val'] =avgf1
            
            print("current f1 %f"%avgf1)


# In[15]:


test = test_df


# In[18]:


from sklearn.model_selection import StratifiedKFold
from keras import backend as K 
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda

# Do some code, e.g. train and save model

oof_test = np.zeros((len(test),1))



config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
bert_model = load_trained_model_from_checkpoint(config_file, checkpoint_file,seq_len=maxlen)#
    #model.summary(line_length=120)
for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x)

model3 = Model([x1_in, x2_in], p)



for i in range(0,5):


    model3.load_weights('../model/best_model_task1_%d.h5'%(i+1))
    
    oof_test += model3.predict(test_x)/5


t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

test_df['label'] = (oof_test>0.5).astype(int)
test_df[['id','label']].to_csv("../submit/submission_%s.csv"%t,index=False)






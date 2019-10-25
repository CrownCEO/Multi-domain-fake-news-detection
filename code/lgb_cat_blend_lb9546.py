#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

import scipy as sp
import pandas as pd
import numpy as np

from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold

from collections import Counter
import os
import sys
import pickle
import gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按照PCI_BUS_ID顺序从0开始排列GPU设备 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'import sys
import lightgbm as lgb
np.random.seed(369)




import scipy as sp

from collections import Counter
from functools import partial
from math import sqrt

from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix


# In[2]:


import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm, tqdm_notebook
from keras.applications.densenet import preprocess_input, DenseNet121


# In[3]:


train = pd.read_csv("../data/task3/train.csv")
test = pd.read_csv("../data/task3/task3_new_stage2.csv")


# In[4]:


fake_pic_dir = os.listdir("../data/task3/train/rumor_pic/")
true_pic_dir = os.listdir("../data/task3/train/truth_pic/")
test_pic_dir = os.listdir("../data/task3/task3_new_stage2_pic/")


os.system("mkdir  ../data/task3/train/all_img")
os.system("cp ../data/task3/train/rumor_pic/*.jpg  ../data/task3/train/all_img/")
os.system("cp ../data/task3/train/truth_pic/*.jpg  ../data/task3/train/all_img/")
os.system("cp ../data/task3/task3_new_stage2_pic/*.jpg  ../data/task3/train/all_img/")

print("创建all_img成功!")

all_images_dir = os.listdir("../data/task3/train/all_img/")



# In[5]:


def pic_is_fake(x):
    x = str(x)
    if x in fake_pic_dir:
        return 1
    if x in true_pic_dir:
        return 2
    return 0

def pic_path(x):
    x = str(x)
    if x in all_images_dir:
        return "./data/task3/train/all_img/"+x
        

    return  np.nan

data = pd.concat([train,test])
data['is_fake_pic'] = data['piclist'].apply(lambda x:pic_is_fake(x))
data['piclist_path'] = data['piclist'].apply(lambda x:pic_path(x))


# In[6]:


data['userLocation2'] = data['userLocation'].fillna("nan").map(lambda x:" " in x)
data['userLocation3'] = data['userLocation'].fillna("nan").map(lambda x:" " not in x and (len(x) > 3))


# In[7]:


# train = data[data.label.notna()]
# test = data[data.label.isna()]
# print(test['userLocation2'].value_counts())
# print(test['userLocation3'].value_counts())


# In[8]:


img_size = 256
batch_size = 256


# In[9]:


def resize_to_square(im):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return new_im

def load_image(path):
    image = cv2.imread(path)
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


# In[10]:


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
from keras.applications.densenet import preprocess_input, DenseNet121


from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
import multiprocessing


def get_img_fea(features):
    inp = Input((256,256,3))
    backbone = DenseNet121(input_tensor = inp, 
                           weights="../model/DenseNet-BC-121-32-no-top.h5",
                           include_top = False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x,axis = -1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:,:,0])(x)

    m = Model(inp,out)
    pic_list = data[data['piclist_path'].notna()]['piclist_path'].tolist()
    n_batches = len(pic_list) // batch_size + 1
    
    from tqdm import tqdm, tqdm_notebook

    import  time
    start = time.time()
    
    success_count =0
    for b in tqdm_notebook(range(n_batches)):
        start = b*batch_size
        end = (b+1)*batch_size
        batch_pic= pic_list[start:end]
        batch_images = np.zeros((len(batch_pic),img_size,img_size,3))
        for i,pic_path in enumerate(batch_pic):
            try:
                batch_images[i] = load_image(pic_path)
                success_count+=1
            except:
                pass
        batch_preds = m.predict(batch_images)
        for i,pic_path in enumerate(batch_pic):
            features[pic_path] = batch_preds[i]

    print(success_count)
    
    

manager = multiprocessing.Manager()
features = manager.dict()
p = multiprocessing.Process(target=get_img_fea,args=(features,) )
p.start()
p.join()


# In[11]:


data_feats = pd.DataFrame.from_dict(features, orient='index')
data_feats.columns = ['pic_'+str(i) for i in range(data_feats.shape[1])]


# In[12]:


data_feats['piclist_path'] = data_feats.index


# In[13]:


data_feats.reset_index(drop=True,inplace=True)


# In[14]:


#merge image features
data = pd.merge(data,data_feats,how='left',on=['piclist_path'])


# In[15]:


data['All_text']=data['text']+" * "+data['userDescription']
data['All_text'].fillna('-1',inplace=True)


# In[16]:


data['all_text_len_num']=data['All_text'].apply(lambda x:len(x))
data['userDescription_len_num']=data['userDescription'].apply(lambda x:len(x) if x is not np.nan else 0)


# In[17]:


import jieba

def cut_words(x):
    
    return ' '.join(jieba.cut(x))


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

n_components = [100,20,50]
text_features = []

text_feas = ['text','userDescription','All_text']

X_text = data[text_feas]


for a,i in enumerate(text_feas):
    # Initialize decomposition methods:
    X_text[i] = X_text[i].astype(str)
    X_text[i] = X_text[i].apply(lambda x:cut_words(x))
    
    
    print('generating features from: {}'.format(i))
    svd_ = TruncatedSVD(n_components=n_components[a], random_state=1337)
    nmf_ = NMF(n_components=n_components[a], random_state=1337)
    lda_ = LatentDirichletAllocation(n_topics=10,random_state=1337,n_jobs=6)
    #pca_ = PCA(n_components=n_components[a], random_state=1993, whiten=True)
    #print(X_text.loc[:, i].values[:10])

    tfidf_col = TfidfVectorizer(min_df=3, ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1).fit_transform(X_text.loc[:, i].values)

    print("SVD")
    svd_col = svd_.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
    print("NMF")
    nmf_col = nmf_.fit_transform(tfidf_col)
    nmf_col = pd.DataFrame(nmf_col)
    nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
    print("LDA")
    lda_col = lda_.fit_transform(tfidf_col)
    lda_col = pd.DataFrame(lda_col)
    lda_col = lda_col.add_prefix('LDA_{}_'.format(i))

    # print("PCA")
    # pca_col = pca_.fit_transform(tfidf_col)
    # pca_col = pd.DataFrame(pca_col)
    # pca_col = pca_col.add_prefix('PCA_{}_'.format(i))

    text_features.append(svd_col)
    text_features.append(nmf_col)
    text_features.append(lda_col)
    
# Combine all extracted features:
text_features = pd.concat(text_features, axis=1)


# In[19]:


data = pd.concat([data,text_features],axis=1)


# In[20]:


drop_feas = ['id','label','piclist', 'piclist_path','is_fake_pic']
features = [i for i in data.columns if i not in drop_feas ]
print(features)


# In[21]:


object_feas = [i for i in features if str(data[i].dtype)=='object']
print(object_feas)


# In[22]:


for fea in object_feas:
    data[fea] = pd.factorize(data[fea])[0]
    data[fea+"_freq"] = data[fea].map(data[fea].value_counts(dropna=False))


# In[23]:




import tqdm

train = data[data.label.notna()]
test = data[data.label.isna()]


data = pd.concat([train, test])

del train, test
gc.collect()


# In[24]:


train = data[data.label.notna()]
test = data[data.label.isna()]


# In[25]:


print(train['label'].value_counts())


# In[26]:


print(len(test))


# In[27]:


features = [i for i in data.columns if i not in drop_feas ]


# In[28]:


print(features)


# In[29]:


#lgb model

import lightgbm as lgb
from sklearn.model_selection import  StratifiedKFold

params =   {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'n_jobs':100,
                    'learning_rate':0.02,
                    'num_leaves': 31,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.4,
                    'subsample_freq':1,
                    'subsample':0.7,
                   # 'n_estimators':10000,
                    'max_bin':255,
                    'verbose':0,
#                    "min_data_in_leaf":80,
                   "lambda_l1" : 0.1,
                   "lambda_l2" : 0.1,
                    'seed': 1993,
                   # 'early_stopping_rounds':200, 
                } 

from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True



# Additional parameters:
early_stop = 200
verbose_eval = 5
num_rounds = 10000
n_splits = 5
NFOLD=5

lgb_oof_train = np.zeros((train.shape[0],1))
lgb_oof_test = np.zeros((test.shape[0],1))
skf = StratifiedKFold(n_splits=5,random_state=4590,shuffle=True)


categorical_columns = object_feas

# for idx, (train_index, valid_index) in enumerate(kf.split(train_df, y, rescuerID)):
    
feature_importance = pd.DataFrame()
score_csv = []
    
    
# for train_index, valid_index in  (skf.split(X_train[features],X_train['AdoptionSpeed'])):
for k,(train_index, valid_index) in  enumerate(skf.split(train[features],train['label'])):

    
    X_tr = train.iloc[train_index, :][features]
    X_val = train.iloc[valid_index, :][features]
    
    y_tr = train.iloc[train_index, :]['label']
    
    y_val = train.iloc[valid_index, :]['label']
    
    
    d_train = lgb.Dataset(X_tr, label=y_tr,categorical_feature=categorical_columns)
    d_valid = lgb.Dataset(X_val, label=y_val)
    watchlist = [d_train, d_valid]
    
    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop,feval=lgb_f1_score)
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    test_pred = model.predict(test[features], num_iteration=model.best_iteration)
    
    lgb_oof_train[valid_index] = val_pred.reshape(-1,1)
    lgb_oof_test += test_pred.reshape(-1,1)/n_splits
    
    print("fold %d f1 is :%f"%(k,f1_score(y_val,(val_pred>0.5).astype(int))))
    score_csv.append(f1_score(y_val,(val_pred>0.5).astype(int)))
    
        
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = k + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)


# In[30]:


#catboost


from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.metrics import log_loss
import lightgbm  as lgb
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import gc
import numpy as np




cat_params = {
'n_estimators': 10000,
'learning_rate': 0.02,
'eval_metric': 'F1',
'loss_function': 'Logloss',
'random_seed': 4590,
'task_type': 'GPU',
'depth': 7,
'early_stopping_rounds': 200

    }

n_num = 1
categorical_features = object_feas
cat_oof_train = np.zeros((len(train), 1))
cat_oof_test = np.zeros((len(test), 1))
feature_importance_df = pd.DataFrame()
score = 0
skf = StratifiedKFold(n_splits=5,random_state=4590,shuffle=True)
folds = skf
splits = folds.split(train, train['label'])
score_csv = []
for i, (tr_idx, val_idx) in enumerate(splits):
    X_tr, X_vl = train[features].iloc[tr_idx], train[features].iloc[val_idx]
    y_tr, y_vl = train['label'].iloc[tr_idx], train['label'].iloc[val_idx]
    clf = CatBoostClassifier(**cat_params)
    clf.fit(
    X_tr, y_tr, cat_features=categorical_features,
    eval_set=(X_vl, y_vl), verbose=100, plot=True)

    y_pred_valid = clf.predict_proba(X_vl)[:, 1]
    cat_oof_train[val_idx] = y_pred_valid.reshape(-1, 1)
    cat_oof_test += clf.predict_proba(test[features])[:, 1].reshape(-1, 1) / folds.n_splits


    score_csv.append(clf.get_best_score()['validation']['F1'])

  

    del X_tr, X_vl, y_tr, y_vl
    # Features imp
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.get_feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    gc.collect()

score = np.mean(score_csv)

print(f"mean score is {score}")


# In[31]:


#blend lgb and catboost
test = pd.read_csv("../data/task3/task3_new_stage2.csv")

test['label'] = lgb_oof_test*0.5+cat_oof_test*0.5
test['label'] = (test['label']>0.4).astype(int)
print(test['label'].value_counts())
#test[['id','label']].to_csv("blend_lgb_cat_2.csv",index=False)


# In[32]:


test['pred'] = lgb_oof_test*0.5+cat_oof_test*0.5


# In[33]:


tmp = test[test['pred']>=0.9]
print(len(tmp))
tmp1 = test[test['pred']<=0.1]
print(len(tmp1))


# In[34]:


add_data = pd.concat([tmp,tmp1])


# In[35]:


add_data.head()


# In[36]:


add_data['label'] = (add_data['pred']>0.5).astype(int)


# In[37]:


print(add_data['label'].value_counts())


# In[38]:


del add_data['pred']
add_data.to_csv("../data/task3/pesudo_data.csv",index=False)


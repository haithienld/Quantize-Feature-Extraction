import cv2
import os
import pandas as pd
import numpy as np
import random
from data.image_preprocess import preprocess,augment_img
# import matplotlib.pyplot as plt
class Group(object):
    def __init__(self,train_df,main_path,image_size,i):
        self.train_df = train_df
        self.source = main_path
        self.members = train_df.loc[train_df['id']==i]
        self.index = i
        self.length_group = 0
        self.image_size = image_size
    def __len__(self):
        self.length_group = len(list(self.members['id']))
        return len(list(self.members['id']))
    def sample(self,match_sample=5,hard_sample=5,non_match=5):

        others = self.train_df.loc[(self.train_df['id']!=self.index)]
        m1 = self.members.sample(match_sample)
        m1 = m1.assign(label=[1]*match_sample)
        m2 = self.members.sample(match_sample)
        m2 = m2.assign(label=[1]*match_sample)

        o = others.sample(non_match+hard_sample)
        o = o.assign(label=[0]*(non_match+hard_sample))
        samples = pd.concat([m1, m2, o], axis=0)
        batch = list()
        batch_label = list()
        for image,label in zip(samples['file'],samples['label']):
            try:
                img = preprocess(os.path.join(self.source,image),self.image_size)
                batch.append(img)
                if label == 1:
                    batch_label.append([0,1])
                if label == 0:
                    batch_label.append([1,0])
            except:
                return [],[]
        return batch, batch_label

    def augment(self,match_sample=5,hard_sample=5,non_match=5):
        folder = list(self.members['folder'])[0]
        same_shape = self.train_df.loc[(self.train_df['folder']==folder)&(self.train_df['id']!=self.index)]
        others = self.train_df.loc[(self.train_df['folder']!=folder)&(self.train_df['id']!=self.index)]
        m = self.members.sample(self.length_group)
        m = m.assign(label=[1]*self.length_group)
        while(self.length_group<=match_sample):
            m.loc[max(m.index+1)] = [random.choice(list(m['file']))] + [0] + [0] +[2]
            self.length_group = self.length_group + 1
        try:
            ss = same_shape.sample(hard_sample)
            ss = ss.assign(label=[0]*hard_sample)
            o = others.sample(non_match)
            o = o.assign(label=[0]*non_match)
            samples = pd.concat([m, ss, o], axis=0)
        except:
            o = others.sample(non_match+hard_sample)
            o = o.assign(label=[0]*(non_match+hard_sample))
            samples = pd.concat([m, o], axis=0)
        batch = list()
        batch_label = list()
        for image,label in zip(samples['file'],samples['label']):
            try:
                if label == 2:
                    img = augment_img(os.path.join(self.source,image),self.image_size)
                # plt.figure(figsize=(20,10))
                # plt.imshow(img)
                # break
                else:
                    img = preprocess(os.path.join(self.source,image),self.image_size)
                batch.append(img)
                if label == 1 or label == 2:
                    batch_label.append([0,1])
                if label == 0:
                    batch_label.append([1,0])
            except:
                return [],[]
        return batch, batch_label

class DataLoader(object):
    def __init__(self,path,embedding_size,image_size,train_list='data.csv'):
        self.main_path = path
        self.train_df = pd.read_csv(train_list, header=None)
        self.train_df.columns = ["file", "id"]
        self.train_df["id"] = self.train_df["id"].apply(pd.to_numeric)
        print(self.train_df)
        self.image_size = image_size
        self.embedding_size = embedding_size

    def get_batches(self,train=True):
        if train == True:
            df = self.train_df
        else:
            df = self.test_df
        used = list()
        while True:
            while(1):
                rand = random.choice(list(set(df['id'])))
                while rand in used:
                    rand = random.choice(list(set(df['id'])))
                used.append(rand)
                groups = Group(df,self.main_path,self.image_size,rand)
                if len(groups) >= 5:
                    batch, batch_label = groups.sample()
                else:
                    continue
                    # batch, batch_label = groups.augment()
                if len(batch) == 0:
                    used.pop()
                else:
                    if len(used)>300:
                        used = list()
                    break
            batches = np.asarray(batch)
            batch_labels = np.asarray(batch_label)
            dummy_label = np.zeros(batches.shape[0])
            yield batches, [dummy_label,batch_labels]

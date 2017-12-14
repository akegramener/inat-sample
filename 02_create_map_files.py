import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from pprint import pprint as pp

files = ['train', 'validation']

class FormatDataForCNTK(object):
    def __init__(self):
        self.cwd = os.getcwd()
        self.metadata = os.path.join(self.cwd, 'metadata')

    def create_map_file(self, path, filename):
        with open(os.path.join(self.metadata, filename), 'wt', encoding='utf-8') as file:
            for label in os.listdir(path):
                for img in os.listdir(os.path.join(path, label)):
                    file.write('{},{}\n'.format(os.path.join(self.cwd, path, label, img), label))

    def store_in_pickle(self, data, filename):
        with open(os.path.join(self.metadata, filename), 'wb') as file:
            pickle.dump(data, file)

    def create_label_id_mapper(self):
        print('Creating label2id and id2label mappers')
        train_data = pd.read_csv(os.path.join(self.metadata, 'train_map.csv'), names=['path', 'label'])
        label2id = dict()
        id2label = dict()
        labels = train_data.label.unique()

        for index, label in enumerate(labels):
            label2id[label] = index
            id2label[index] = label

        self.store_in_pickle(label2id, 'label2id')
        self.store_in_pickle(id2label, 'id2label')

    def add_id_map_file(self, filename, label2id):
        map_file = pd.read_csv(os.path.join(self.metadata, filename), names=['path', 'label'])
        map_file['id'] = map_file['label'].map(label2id)
        map_file.to_csv(os.path.join(self.metadata, filename), header=False, index=False)

    def create_map_files(self):
        print('Creating csv files from the directory')
        for file in files:
            self.create_map_file('{}/{}/'.format('data', file), '{}_map.csv'.format(file))

    def store_id_map_files(self):
        print('Adding id to the csv files')
        label2id = pickle.load(open(os.path.join(self.metadata, 'label2id'), 'rb'))
        for file in files:
            self.add_id_map_file('{}_map.csv'.format(file), label2id)

    def get_class_weights(self, y, smooth_factor=0.15):
        counter = Counter(y)
        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p
        majority = max(counter.values())
        return {cls: int(majority // count) for cls, count in counter.items()}

    def balance_weight(self, map_file):
        data = pd.read_csv(map_file, names=['path', 'label', 'id'])
        print('\n\nBefore Balancing the weights in {}: \n'.format(map_file))
        pp(data['id'].value_counts())

        copy = data.copy()
        d = self.get_class_weights(data['id'].values)
        for id, count in d.items():
            count -= 1
            if count:
                subset = data[data['id'] == id]
                copy = copy.append([subset] * count, ignore_index=True)
        print('\n\nAfter Balancing the weights in {}: \n'.format(map_file))
        pp(copy['id'].value_counts())
        copy.to_csv(map_file, header=False)

    def balance_weights(self):
        print('Balancing the weights')
        self.balance_weight('metadata/train_map.csv')
        # Don't need to balance the test & validation weights
        # self.balance_weight('metadata/{}_map.csv')

    def create_input_cntk_map(self, file):
        src = os.path.join(self.metadata, file)
        dest = os.path.join(self.metadata, os.path.splitext(file)[0] + '.txt')
        data = pd.read_csv(src, names=['path', 'label', 'id'])
        map_format = data[['path', 'id']]
        map_format.to_csv(dest, index=False, header=False, sep='\t')

    def create_input_cntk_maps(self):
        print('Creating input map text files')
        for file in files:
            self.create_input_cntk_map('{}_map.csv'.format(file))

    def structure_data(self):
        print('Starting...')
        self.create_map_files()
        self.create_label_id_mapper()
        self.store_id_map_files()
        self.balance_weights()
        self.create_input_cntk_maps()


if __name__ == '__main__':
    if not os.path.exists('metadata'):
    	os.makedirs('metadata')
    ctf = FormatDataForCNTK()
    ctf.structure_data()

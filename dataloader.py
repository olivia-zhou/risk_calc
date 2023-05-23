"""
DATALOADER:
extracts, cleans, and prepares data

inputs: datalist
outputs: randomly shuffled and split train and validation datasets

CLASSES:
*data: opens csv file, splits data into train and validation datasets

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

class data():
    def __init__(self, features, labels, csv):
        super(data, self).__init__()
        self.features = features
        self.label = labels
        content = self.read_csv(csv)
        self.content = self.remove_missing(content)
        self.labels = self.get_labels
        self.train, self.valid = self.split_train_valid_set(self.content)
    
    def read_csv(self, csv_dict):
        content = []
        with open(csv_dict, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content.append(row)
        return content

    def split_train_valid_set(self, data):
        np.random.shuffle(data)
        train, valid = data[:70,:], data[70:,:]
        return train, valid

    def remove_missing(self, content):
        filtered_content = []
        for row in content:
            complete = True
            for key in self.features:
                if row[key] == '':
                    complete = False
            if complete and row[self.label] != '':
                filtered_content.append(row)
        return filtered_content
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        row = self.content[idx]
        label = int(row[self.label])
        name = row['ID'] # the name value will be used as ID when caching model predictions
        data = []
        for feature in self.features:
            if feature == 'gender':
                data.extend(self._onehot_gender(row[feature]))
            elif feature == 'race':
                data.extend(self._onehot_race(row[feature]))
            else:
                data.append(float(row[feature]))
        data = np.array(data)
        return data, label, name
    
    def get_labels(self):
        labels = []
        for row in self.content:
            labels.append(int(row[self.label]))
        return labels
    
    def _onehot_gender(self, gender):
        if gender == 'male':
            return [0, 1]
        elif gender == 'female':
            return [1, 0]
        else:
            raise ValueError("gender value not found")

    def _onehot_race(self, race):
        if race == 'whi':
            return [0, 0, 0, 0, 0, 1]
        elif race == 'blk':
            return [0, 0, 0, 0, 1, 0]
        elif race == 'ind':
            return [0, 0, 0, 1, 0, 0]
        elif race == 'haw':
            return [0, 0, 1, 0, 0, 0]
        elif race == 'ans':
            return [0, 1, 0, 0, 0, 0]
        elif race == 'mix':
            return [1, 0, 0, 0, 0, 0]
        else:
            raise ValueError("race value not found")





if __name__ == '__main__':
    print('test')
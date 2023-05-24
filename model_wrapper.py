"""
MODEL WRAPPER:
sets up and trains the model

input: 
output:
"""
import pytorch_tabnet
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import matplotlib.pyplot as plt
from dataloader import data

import os, json
import numpy as np

class wrapper(object):

    def __init__(self, experiment_name, device, data_settings, hyper_params):
        super(wrapper, self).__init__()

        # --------------------------------------------------------------------------------------------------------------
        # general settings and hyperparameters
        self.experiment_name = experiment_name
        self.device = device
        self.data_path = data_settings['data_path']
        self.features = data_settings['features']
        self.labels = data_settings['labels']
        self.epochs = hyper_params['max_epochs']
        self.loss_type = hyper_params['loss_type']
        self.batch_size = hyper_params['batch_size']
        self.cat_idxs = hyper_params['cat_idxs']
        self.cat_dims = hyper_params['cat_dims']
        self.lr = hyper_params['lr']

        # --------------------------------------------------------------------------------------------------------------
        # folders preparation to save checkpoints of model weights *.pth
        self.root_dir = './experiments/'
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
        self.exp_dir = './experiments/{}/'.format(self.experiment_name)
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        # --------------------------------------------------------------------------------------------------------------
        #load data
        train_data, valid_data = data(self.features, self.labels, self.data_path)
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size,
                                  drop_last=True, shuffle=True) #sampler=SubsetRandomSampler(train_data.__len__()), num_workers=self.num_workers, 
        self.valid_loader = DataLoader(valid_data, batch_size=self.batch_size, 
                                  drop_last=True, shuffle=False) #sampler=SubsetRandomSampler(valid_data.__len__()), num_workers=self.num_workers, 
        print("dataloader initialized")

        # --------------------------------------------------------------------------------------------------------------
        #set up model
        self.model = TabNetMultiTaskClassifier(n_steps=1,
                                        cat_idxs=self.cat_idxs,
                                        cat_dims=self.cat_dims,
                                        cat_emb_dim=1,
                                        optimizer_fn=torch.optim.Adam,
                                        optimizer_params=dict(lr=self.lr),
                                        scheduler_params={"step_size":50, # how to use learning rate scheduler
                                                  "gamma":0.9},
                                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                        mask_type='entmax', # "sparsemax",
                                        lambda_sparse=0, # don't penalize for sparser attention
                    )
        
    # --------------------------------------------------------------------------------------------------------------
    # train model
    def train(self):
        max_epochs = self.epochs
        self.model.fit(X_train=x_train, y_train=y_train,
                    eval_set = [(x_valid, y_valid)],
                    patience = 50, #T_T a long time
                    max_epochs=max_epochs,
                    batch_size=1024,
                    virtual_batch_size=128,
                    num_workers=1,
                    drop_last=True
        )
        # plot losses - drop first epoch
        plt.plot(self.model.history['train']['loss'][5:])
        plt.plot(self.model.history['valid']['loss'][5:])
        plt.plot([x for x in self.model.history['train']['lr']][5:])

    # --------------------------------------------------------------------------------------------------------------
    # validate model
    def validation(self):
        preds_valid = self.model.predict_proba(x_valid) # This is a list of results for each task
        # We are here getting rid of tasks where only 0 are available in the validation set
        valid_aucs = [roc_auc_score(y_score=task_pred[:,1], y_true=y_valid[:, task_idx])
                    for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]

        valid_logloss = [log_loss(y_pred=task_pred[:,1], y_true=y_valid[:, task_idx])
                    for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]

        plt.scatter(y_valid.sum(axis=0)[y_valid.sum(axis=0)>0], valid_aucs)

        print(f"BEST VALID SCORE FOR {self.experiment_name} : {self.model.best_cost}")
        print(f"VALIDATION MEAN LOGLOSS SCORES FOR {self.experiment_name} : {np.mean(valid_logloss)}")
        print(f"VALIDATION MEAN AUC SCORES FOR {self.experiment_name} : {np.mean(valid_aucs)}")

    # --------------------------------------------------------------------------------------------------------------
    # model outputs
    def get_predictions(self):
        preds = self.model.predict_proba(x_test)
        return preds
    
    # --------------------------------------------------------------------------------------------------------------
    # save weights
    def save_model(self):
        name = "./model_test_{}".format(self.experiment_name)
        save_path = self.exp_dir + name
        saved_filepath = self.model.save_model(save_path)
        print('successfully saved model weights')

    # --------------------------------------------------------------------------------------------------------------
    # load checkpoint
    def load_checkpoint(self, filepath):
        loaded_model = TabNetMultiTaskClassifier()
        loaded_model.load_model(filepath)
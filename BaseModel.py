from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Base_Model():
    def __init__(self, train_df, test_df, feats,target, categorical_variables = [],
                 parms_dict = {}, rm_feats = [], verbose = True, kfold = True,
                 folds = 5, val_in_train = False, val_indicator_col = 'is_valid',
                 kf = KFold(n_splits=5, random_state = 11, shuffle = True), no_val = False,
                 target_categorical=False):
        self.train = train_df
        self.test = test_df
        self.feats = feats
        self.target = target
        self.categorical_variables = categorical_variables   
        self.parms_dict = parms_dict
        self.rm_feats = rm_feats
        self.verbose = verbose
        self.kfold = kfold 
        self.folds = folds   
        self.val_in_train = val_in_train
        self.val_indicator_col = val_indicator_col
        self.kf = kf
        self.no_val = no_val
        self.test_preds, self.models, self.val_preds = [], [], None
        self.target_categorical = target_categorical
        
    def convert_x(self,x):
        return x
    
    def convert_dataset(self, x, y):
        raise NotImplementedError
        
    def fit(self, train_set, val_set, test_set):
        raise NotImplementedError
    
    def loss(self, y_true, y_pred):
        if self.target_categorical:
            return log_loss(y_true, y_pred)
        else:
            return mean_squared_error(y_true, y_pred)      
            
        
    def plot_feature_importance(self):
        raise NotImplementedError

    def train_model(self):
        
        self.feats = [f for f in self.feats if f not in self.rm_feats]
        if self.no_val:
            X_train = self.train[self.feats]
            Y_train = self.train[self.target]

            X_test = self.test[self.feats]
            train_set = self.convert_dataset(X_train, Y_train)
            test_set = self.convert_x(X_test)
            test_pred, val_pred,model = self.fit(train_set,None,test_set,None,None,X_test)
            self.test_preds.append(test_pred)
            self.models.append(model)

            
        if self.val_in_train:
            X_train = self.train.loc[self.train[self.val_indicator_col]==False,self.feats].reset_index(drop = True)
            Y_train = self.train.loc[self.train[self.val_indicator_col]==False,self.target].reset_index(drop = True)
            
            X_val = self.train.loc[self.train[self.val_indicator_col]==True,self.feats].reset_index(drop = True)       
            Y_val = self.train.loc[self.train[self.val_indicator_col]==True,self.target].reset_index(drop = True)
            
            X_test = self.test[self.feats]
            
            train_set, val_set = self.convert_dataset(X_train, Y_train), self.convert_dataset(X_val, Y_val)
            test_set = self.convert_x(X_test)
            
            test_pred, val_pred,model = self.fit(train_set,val_set,test_set,X_val,Y_val,X_test)
            
            self.test_preds.append(test_pred)
            self.val_preds = val_pred
            self.models.append(model)
            
        if self.kfold == True:
            self.val_preds = np.zeros(self.train.shape[0])
            for train_idx, val_idx in self.kf.split(self.train, self.train[self.target]):
                X_train = self.train.loc[train_idx, self.feats].reset_index(drop = True)
                Y_train = self.train.loc[train_idx,self.target].reset_index(drop = True)

                X_val = self.train.loc[val_idx, self.feats].reset_index(drop = True)       
                Y_val = self.train.loc[val_idx,self.target].reset_index(drop = True)

                X_test = self.test[self.feats]
                
                train_set, val_set = self.convert_dataset(X_train, Y_train), self.convert_dataset(X_val, Y_val)
                test_set = self.convert_x(X_test)

                test_pred, val_pred,model = self.fit(train_set,val_set,test_set,X_val,Y_val,X_test)
                
                self.test_preds.append(test_pred)
                self.val_preds[val_idx] = val_pred
                self.models.append(model)
        return self.test_preds, self.val_preds, self.models
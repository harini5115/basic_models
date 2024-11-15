from My_models.BaseModel import Base_Model
from lightgbm import LGBMClassifier
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class lightgbmClassifier(Base_Model):
    def plot_importance(self):
        imp_df = pd.DataFrame()
        imp_df['feats'] = self.feats
        importances = []
        for model in self.models:
            importances.append(model.feature_importances_)
        importance = np.sum(importances, axis = 0)
        imp_df['gain'] = importance
        imp_df = imp_df.sort_values('gain', ascending = False)
        plt.figure(figsize = (8,8))
        sns.barplot(x = 'gain', y = 'feats', data = imp_df)
        plt.show()
        return imp_df
    
    def convert_dataset(self, x_train, y_train):
        return (x_train,y_train)
    
    def fit(self,train_set, val_set, test_set,X_val,Y_val,X_test):
        verbosity = 100 if self.verbose else 0 
        val_pred = None
        if val_set:
            val_values = Y_val
            clf = LGBMClassifier().set_params(**self.parms_dict)
            early_stopping = lgb.early_stopping(stopping_rounds=self.parms_dict['early_stopping_rounds'])
            log_evaluation = lgb.log_evaluation(period=100)

            clf.fit(train_set[0], train_set[1], 
                    eval_set= [train_set, val_set], 
                    eval_metric=['auc', 'logloss'],
                    callbacks=[early_stopping, log_evaluation],
                    categorical_feature=self.categorical_variables,
                   )

            val_pred = clf.predict_proba(X_val, num_iteration=clf.best_iteration_)[:, 1]
            print(f'loss is :{self.loss(val_values, val_pred)}')
        else:
            clf = LGBMClassifier().set_params(**self.parms_dict)
            log_evaluation = lgb.log_evaluation(period=100)
            clf.fit(train_set[0], train_set[1], 
                    eval_set= [train_set], 
                    eval_metric=['auc', 'logloss'],
                    callbacks=[log_evaluation],
                    categorical_feature=self.categorical_variables,
                   )            
        test_pred = clf.predict_proba(self.convert_x(X_test),num_iteration=clf.best_iteration_)[:,1]
        return test_pred,val_pred, clf
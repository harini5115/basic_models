from My_models.BaseModel import Base_Model
from catboost import CatBoostRegressor
import seaborn as sns

class catBoostRegressor(Base_Model):
    
    def plot_importance(self):
        imp_df = pd.DataFrame()
        imp_df['feats'] = self.feats
        importances = []
        for model in self.models:
            importances.append(model.feature_importances_)
        importance = np.sum(importances, axis = 0)
        imp_df['gain'] = importance
        imp_df = imp_df.sort_values('gain', ascending = False)
        plt.figure(figsize=(8,8))
        sns.barplot(x = 'gain', y = 'feats', data = imp_df)
        plt.show()
        return imp_df
        
    def convert_dataset(self,x_train, y_train):
        return {'X': x_train, 'y':y_train}
    
    def fit(self, train_set,val_set,test_set,X_val,Y_val,X_test):
        verbosity = 100 if self.verbose==True else 0
        catReg = CatBoostRegressor(**self.parms_dict)
        val_pred = None
        if val_set:
            val_values = val_set['y']
            catReg.fit(train_set['X'], train_set['y'],
                       eval_set = (val_set['X'], val_set['y']),
                       cat_features = self.categorical_variables,
                       verbose=verbosity)
            val_pred = catReg.predict(self.convert_x(val_set['X']))
            print(f'loss is :{self.loss(val_values, val_pred)}')
        else:
            catReg.fit(train_set['X'], train_set['y'],cat_features = self.categorical_variables,
                       verbose=verbosity)    
        test_pred = catReg.predict(test_set)
        return test_pred, val_pred, catReg
    
    
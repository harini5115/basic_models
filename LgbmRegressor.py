from My_models.BaseModel import Base_Model
import lightgbm as lgb
import seaborn as sns

class lightgbmRegressor(Base_Model):
    def plot_importance(self):
        imp_df = pd.DataFrame()
        imp_df['feats'] = self.feats
        importances = []
        for model in self.models:
            importances.append(model.feature_importance())
        importance = np.sum(importances, axis = 0)
        imp_df['gain'] = importance
        imp_df = imp_df.sort_values('gain', ascending = False)
        plt.figure(figsize = (8,8))
        sns.barplot(x = 'gain', y = 'feats', data = imp_df)
        plt.show()
        return imp_df
    
    def convert_dataset(self, x_train, y_train):
        return lgb.Dataset(x_train,y_train,categorical_feature=self.categorical_variables,free_raw_data=False)
    
    def fit(self,train_set, val_set, test_set,X_val,Y_val,X_test):
        verbosity = 100 if self.verbose else 0 
        val_pred = None
        if val_set:
            val_values = Y_val
            lgbReg = lgb.train(self.parms_dict, train_set, valid_sets=[train_set, val_set],
                               valid_names=['train', 'val'],categorical_feature=self.categorical_variables,
                               verbose_eval=verbosity)
            val_pred = lgbReg.predict(self.convert_x(X_val))
            print(f'loss is :{self.loss(val_values, val_pred)}')
        else:
            lgbReg = lgb.train(self.parms_dict, train_set, valid_sets=[train_set],
                   valid_names=['train'],categorical_feature=self.categorical_variables,
                   verbose_eval=verbosity)
        test_pred = lgbReg.predict(self.convert_x(X_test))
        return test_pred,val_pred, lgbReg

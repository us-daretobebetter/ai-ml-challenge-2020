
#classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
import tensorflow as tf


class ML_classifiers:
    classifiers = {
        # "KNN":{
        #     'classifier':lambda self:KNeighborsClassifier(),
        #     'params':{
        #         'n_neighbors':5,#10,20
        #         'weights':'distance'#'uniform',
        #     }
        # },
        # "SVC":{
        #     'classifier':lambda self: SVC(),
        #     'params':{
        #         'kernel':'linear',#'rbf',,'sigmoid'
        #         'gamma' :'auto'#,'scale'
        #     }
        # },
        # "Decision Tree":{
        #     'classifier':lambda self:DecisionTreeClassifier(),
        #     'params':{}
        # },
        "Random Forest":{
            'classifier':lambda self:RandomForestClassifier(random_state=127),
            'params':{
                'criterion':'gini',#'entropy'],
                'max_features':'auto',#,'log2','sqrt']
                 'n_estimators':1000,
                 'max_depth': 2000
            }
        },
        # "MLP":{
        #     'classifier':lambda self: MLPClassifier(random_state=127),
        #     'params':{
        #         'activation':'relu',#,'logistic','tanh'
        #         'solver':'adam',#,'sgd','lbfgs'
        #         'learning_rate':'adaptive'#'constant',
        #     }
        # },
        # "AdaBoost":{
        #     'classifier':lambda self:AdaBoostClassifier(random_state=127),
        #     'params':{}
        # },
        # "XGBoost":{
        #     'classifier':lambda self:XGBClassifier(),
        #     'params':{}
        # },
        # "Naive Bayes":{
        #     'classifier':lambda self:GaussianNB(),
        #     'params':{}
        # },
        # "Logistic Regression":{
        #     'classifier':lambda self: LogisticRegression(random_state=127),
        #     'params':{
        #         #'penalty':[]#'l1','l2','elasticnet'
        #     }
        # },
    }


    def __init__(self):
        return


    def build_clf(self,clf_name):
        """
        :param clf_name: clarify classifier name
        :return: classifier model, its parameters

        """
        model = self.classifiers[clf_name]
        #print(model)
        return model['classifier'](self),model['params']



    def clf_performance(self,cm_ravel,best_params,beta=1):
        """

        :param cm_ravel: confusion matrix.ravel for the specified classifier
        :return: corresponding model performance
        #in main: a nested dict with clf_name as the upper-level key, corresponding model performance dict as value

        """
        tn, fp, fn, tp = cm_ravel

        clf_metrics={}

        clf_metrics['recall'] = tp / (tp + fn)
        clf_metrics['precision'] = tp / (tp + fp)
        clf_metrics['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
        clf_metrics['f_score'] = (1 + beta ** 2) * (clf_metrics['recall'] * clf_metrics['precision']) / (
                    clf_metrics['recall'] + beta ** 2 * clf_metrics['precision'])
        clf_metrics['true_positive'] = tp
        clf_metrics['true_negative'] = tn
        clf_metrics['false_positive'] = fp
        clf_metrics['false_negative'] = fn
        #add params
        clf_metrics['best_params'] = best_params


        return clf_metrics







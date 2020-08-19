#preprocess for modelling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter





class Preprocess():

    def __init__(self,X,y):
        self.X = X
        self.y = y
        #pass
        return


    def resampling(self, oversample_ratio=0.3, minority_num=368, majority_num=10000,minority_label='1.0',majority_label='0.0'):
        # define resampling
        under = RandomUnderSampler(sampling_strategy={majority_label: majority_num, minority_label: minority_num})
        over = SMOTE(sampling_strategy=oversample_ratio)

        # define pipeline
        pipeline = Pipeline(steps=[('u', under), ('o', over)])

        X_sm, y_sm = pipeline.fit_resample(self.X, self.y)

        print('Proportion in data after resample: ',Counter(y_sm))

        return X_sm, y_sm


    def split_and_scale(self,X_sm,y_sm,test_size=0.2,random_state=127):
        """

        :param X_sm: X after resampling
        :param y_sm: y after resampling
        :param test_size:
        :param random_state:
        :return: X_train and X_valid in array
        """
        X_train, X_valid, y_train, y_valid = train_test_split(X_sm, y_sm, test_size=test_size, random_state=random_state)

        # Standardize data
        scaler = StandardScaler()

        # get array
        scaler1 = scaler.fit(X_train)
        X_train_scale = scaler1.transform(X_train)
        X_valid_scale = scaler1.transform(X_valid)

        print('Completed Data Split!')
        return X_train_scale, X_valid_scale, y_train, y_valid,scaler1

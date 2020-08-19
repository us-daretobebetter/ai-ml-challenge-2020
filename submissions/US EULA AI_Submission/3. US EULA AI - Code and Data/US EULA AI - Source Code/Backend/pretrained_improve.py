import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, brier_score_loss
from collections import Counter
from modules import nlp,preprocess,models,text_augmentation
#import pickle


# TODO: setting for training model
upload_data = 'data/'
model_collection_name = 'model'
tfidf_features_num = 5000
oversample_size = 1 #0.5 0.7 0.5+116
training_test_size = 0.3
perf_path = 'tmp_model_perf/'
random_seed =116  #127  116 109
unseen_data_size=0.1 #0.15 0.20
number_of_synonym = 3 #3 5



print('No',model_collection_name,' has been found. Training model starts.')
#training_data = pd.read_csv(upload_data+'AI_ML_Challenge_Training_Data_Set_1_v1.txt',sep=',')
training_data = pd.read_csv(upload_data+'AI_ML_Challenge_Training_Data_Set_1_v1.txt',sep=',')
appendix= pd.read_excel(upload_data+'Clauses_From_Appendix.xlsx')
training_data = pd.concat([training_data,appendix],axis=0).reset_index(drop=True)
training_data['Clause ID'] = training_data['Clause ID'].fillna(training_data[training_data['Clause ID'].isna()]['Clause ID'].index.to_series())

#training_data.columns
#transform type
training_data['Classification'] = training_data['Classification'].astype(float).astype(str)
# drop nan values
training_data.drop(training_data[training_data['Classification'] == 'nan'].index, inplace=True)
# preprocess data
train_df = nlp.nlp_cleaning_pre(training_data, colname='PRE_CLEAN_TEXT', textcol='Clause Text')
#train_df.to_csv(upload_data+'updated_nlp_cleaning_data.csv',index=False)
#train_df=pd.read_csv(upload_data+'updated_nlp_cleaning_data.csv')


#TODO: get combined df (train_df + augmented data) first
#train_df = combined_df.copy()
#train_df.isnull().sum()
# train_df[train_df.isna().any(axis=1)].index
#train_df.drop(train_df[(train_df['Classification'] == 'nan')|(train_df.isna().any(axis=1))].index, inplace=True)
#train_df['Classification'] = train_df['Classification'].astype(float).astype(str)

training_x = train_df.loc[:,train_df.columns!='Classification']
training_y = train_df['Classification']



train_df_x, unseen_x, train_df_y, unseen_y = train_test_split(training_x, training_y,
                                                    test_size=unseen_data_size,
                                               random_state=random_seed)


train_df = pd.concat([train_df_x,train_df_y],axis=1)
#train_df.to_csv('real_training_data_75pc.csv',index=False)
#train_df = pd.read_csv('real_training_data_75pc.csv')
mock_unseen = pd.concat([unseen_x,unseen_y],axis=1)
#mock_unseen.to_csv('real_unseen_data_15pc.csv',index=False)
#mock_unseen =  pd.read_csv('real_unseen_data_15pc.csv')
#mock_unseen.drop(mock_unseen[(mock_unseen['Classification'] == 'nan')|(mock_unseen.isna().any(axis=1))].index, inplace=True)
#mock_unseen['Classification'] = mock_unseen['Classification'].astype(float).astype(str)


#get augmented text
#train_df = generate_augmented_text(base_df=train_df,common_cols = ['Clause ID', 'Classification', 'PRE_CLEAN_TEXT'])
train_df,aug_df = text_augmentation.get_augmented_df(base_df=train_df,common_cols = ['Clause ID', 'Classification', 'PRE_CLEAN_TEXT'],number_of_synonym = number_of_synonym)
#Counter(train_df['Classification'])



#save transformer
tfidf = TfidfVectorizer(max_features=tfidf_features_num)
print(type(train_df))

save_vectorizer = tfidf.fit(train_df['PRE_CLEAN_TEXT'])
train_features = save_vectorizer.transform(train_df['PRE_CLEAN_TEXT'])
train_target = train_df['Classification']
#save_vectorizer.get_feature_names()

#train model
print('Data ready to fit into models.')

prep = preprocess.Preprocess(train_features, train_target)

# oversample
X_sm, y_sm = prep.resampling(oversample_ratio=oversample_size,
                             minority_num=train_df['Classification'].value_counts()[1],
                             majority_num=train_df['Classification'].value_counts()[0],
                             minority_label=train_df['Classification'].value_counts().index[1],
                             majority_label=train_df['Classification'].value_counts().index[0])



Counter( y_sm)

#save_df = ndarray_to_df(x_ndarray = X_sm.toarray(),x_columns=save_vectorizer.get_feature_names(),x_index=len(X_sm.toarray()),y_ndarray=y_sm,y_columns=['Classification'],y_index=len(y_sm))

#save_df.to_csv(upload_data+'updated_nlp_smote_augmented_80PC_data.csv',index=False)
#save_df = pd.read_csv(upload_data+'updated_nlp_smote_augmented_80PC_data.csv')
#save_df.shape
#save_df.columns[:20]


model_class = models.ML_classifiers()
clf_names = list(model_class.classifiers.keys())
print('Pre-determined models:',clf_names)
X_train, y_train = X_sm.toarray(), y_sm

X_test = save_vectorizer.transform(mock_unseen['PRE_CLEAN_TEXT']).toarray()
y_test = mock_unseen['Classification']

# TODO:commented out the original training code
# X_train, X_test, y_train, y_test = train_test_split(X_sm.toarray(), y_sm,
#                                                     test_size=training_test_size,
#                                                random_state=127)

#TODO: no resampling on combined_df
# X_train, X_test, y_train, y_test = train_test_split(train_features, train_target,
#                                                     test_size=training_test_size,
#                                                random_state=127)


perf={}
clf_names = [e for e in clf_names if e not in ['SVC']]
for clf in clf_names:
    #clf = 'MLP'
    clf_model,clf_params = model_class.build_clf(clf)
    #clf_model = GridSearchCV(clf_model, clf_params,cv=5)
    clf_model.set_params(**clf_params)
    best_model = clf_model.fit(X_train,y_train)
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:,1]
    clf_perf= model_class.clf_performance(confusion_matrix(y_test,y_pred).ravel(),
                                         best_params =clf_params)#use [best_model.best_params_] for gridsearchCV )
    #print(best_model.classes_)
    pred_proba_positive = best_model.predict_proba(X_test)[:,1]
    #add brier score to clf_perf dict
    clf_perf['brier_score'] = brier_score_loss(y_test,y_pred_prob,pos_label=str(train_df['Classification'].value_counts().index[1]))


    perf[clf] = clf_perf
print('Finish initial training.')
print('best parameter',clf_params)#use [best_model.best_params_] for gridsearchCV )
#tmp
# beta=1
# clf_metrics={}
# tn, fp, fn, tp  = confusion_matrix(y_test,y_pred).ravel()
#
# clf_metrics['recall'] = tp / (tp + fn)
# clf_metrics['precision'] = tp / (tp + fp)
# clf_metrics['accuracy'] = (tp+tn)/(tp+tn+fp+fn)
# clf_metrics['f_score'] = (1 + beta ** 2) * (clf_metrics['recall'] * clf_metrics['precision']) / (clf_metrics['recall'] + beta ** 2 * clf_metrics['precision'])
#tmp end

print('brier_score',clf_perf['brier_score'])
result = pd.DataFrame.from_dict(perf,orient='index').sort_values(by='brier_score',ascending=False)
result =result[['brier_score','f_score','recall', 'precision', 'accuracy','true_positive',
       'true_negative', 'false_positive', 'false_negative', 'best_params']]

result.to_csv(perf_path+'RFGrid'+str(unseen_data_size)+'PC_real_unseen_resample'+str(oversample_size)+'_aug'+str(number_of_synonym)+'_'+str(random_seed)+'.csv')




# #use all data to do training in practice
# X_train_final = X_sm.toarray()
# y_train_final = y_sm
#
# best_model = list(model_class.classifiers.keys())[0]
# clf_model, clf_params = model_class.build_clf(best_model)
#
# best_model = clf_model.set_params(**clf_params)
# best_model.fit(X_train_final, y_train_final)

#     # save trained model to db
#     db.save_model_to_db(model=save_model,
#                         model_name=best_model,
#                         vectorizer=save_vectorizer,
#                         feature_include = [],
#                         feature_exclude =[],
#                         performance=[],
#                         best_params= [clf_params],
#                         collection=model_collection_name)
#     print('Saved the best model to db.')
# else:
#     print('Please only select the best model ')
#
# elif if_exist==1:
#     print('Found previous model in db to use!')
#
#     #read model and saved_vectorizer out
#     read_model_json = db.load_saved_data_from_db(collection=model_collection_name)
#
#
#     save_model = pickle.loads(read_model_json['model'])
#     save_vectorizer = pickle.loads(read_model_json['vectorizer'])
#
#
#
#
#
# #
# testing_df = pd.read_csv('C:/Users/xliu/Downloads/GSA_EULA_Challenge/DATA/airbus_unseen.csv')
# testing_df = nlp.nlp_cleaning_pre(testing_df, colname='PRE_CLEAN_TEXT', textcol='Clause Text')
#
# testing_features = save_vectorizer.transform(testing_df['PRE_CLEAN_TEXT'])
# testing_pred_class = best_model.predict(testing_features)
# testing_pred_prob = best_model.predict_proba(testing_features)[:,1]
# Counter(testing_pred_class)
#
#
# test_pred_df=testing_df.copy()
# test_pred_df['Predicted_Label'] = pd.Series(testing_pred_class)
# test_pred_df['Predicted_Label_Proba'] = pd.Series(testing_pred_prob).round(4)
# test_pred_df.to_csv('xxxresults.csv',index=False)
#
# db.raw_data_to_db(test_pred_df[['Clause ID','Predicted_Label','Predicted_Label_Proba']],collection='predict_results')
#
#
#






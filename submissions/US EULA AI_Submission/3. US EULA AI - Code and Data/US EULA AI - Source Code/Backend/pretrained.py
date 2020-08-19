import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from modules import mongodb,nlp,preprocess,models
from modules import text_augmentation as aug

# TODO: setting for training model
upload_data = 'data/'
tempFolder = 'temp_file/'
complied_model = 'model/'
model_collection_name = 'model'
tfidf_features_num = 5000
oversample_size = 1 #0.5
synonym_number = 3
model_class = models.ML_classifiers()
best_model_name = list(model_class.classifiers.keys())[0]
vec_name =  'TfidfVectorizer'
save_ml = complied_model + best_model_name + '.pkl'
save_vec = complied_model + vec_name + '.pkl'

# TODO: set path for the original gsa eula data
training_data = pd.read_csv(upload_data+'AI_ML_Challenge_Training_Data_Set_1_v1.txt',sep=',')
appendix= pd.read_excel(upload_data+'Clauses_From_Appendix.xlsx')
training_data = pd.concat([training_data,appendix],axis=0).reset_index(drop=True)
training_data['Clause ID'] = training_data['Clause ID'].fillna(training_data[training_data['Clause ID'].isna()]['Clause ID'].index.to_series())
common_cols = ['Clause ID', 'Classification', 'PRE_CLEAN_TEXT']
gsa_eula = training_data.copy()




def pretrain(db_client = 'localhost:27017/', db_name= 'GSA_EULA',if_retrain='no',combined_df=None):

    db = mongodb.save_read_db(client=db_client, db=db_name)
    #check db if model exists
    if_exist = db.check_if_exists(collection_name=model_collection_name,model_path =save_ml ,vec_path =save_vec )


    if if_exist == 1 and if_retrain == 'no':
        print('Found previous model in db to use!')

        # read model and saved_vectorizer out
        save_vectorizer = db.read_gridfs_data(vec_name, if_df='no', collection='model')
        print('read',type(save_vectorizer))
        save_model =db.read_gridfs_data(best_model_name,if_df='no',collection ='model')
        print('read',type(save_model))

    else:


        if if_retrain=='no'and if_exist == 0:

                print('No',model_collection_name,' has been found. Training model starts.')

                print('Start text augmentation process!')
                training_data,aug_df = aug.get_augmented_df(base_df=gsa_eula,common_cols=common_cols,number_of_synonym = synonym_number)
                training_data.rename(columns={'PRE_CLEAN_TEXT': 'Clause_Text',
                                              'Clause Text': 'Clause_Text',
                                              'Clause ID': 'Clause_Id',
                                              'Full Clause': 'Full_Clause',
                                              'Doc_id': 'Doc_Id'}, inplace=True)
                print('Completed text augmentation process!')

        elif if_retrain =='yes' and if_exist ==1:
                #load previous training data
                training_data = pd.read_csv(upload_data+'raw_training.csv')

                print('previous data',training_data.shape)
                print('updated', combined_df.shape)
                training_data = pd.concat([training_data,combined_df],axis=0).drop_duplicates(subset=['Clause_Text'],keep='last')
                training_data.reset_index(inplace=True,drop=True)
                print('updated total', training_data.shape)
                print('Successfully combined previous training data and user verified cases!')
        #transform type
        training_data['Classification'] = training_data['Classification'].astype(float).astype(str)
        # drop nan values
        training_data.drop(training_data[training_data['Classification'] == 'nan'].index, inplace=True)
        #save training
        training_data.to_csv(upload_data+'raw_training.csv',index=False)
        # preprocess data
        train_df = nlp.nlp_cleaning_pre(training_data, colname='Pre_Clean_Text', textcol='Clause_Text')
        print('Completed NLP.')

        #save transformer
        tfidf = TfidfVectorizer(max_features=tfidf_features_num)

        save_vectorizer = tfidf.fit(train_df['Pre_Clean_Text'])
        train_features = save_vectorizer.transform(train_df['Pre_Clean_Text'])
        train_target = train_df['Classification']

        #train model
        print('Data ready to fit into models.')

        prep = preprocess.Preprocess(train_features, train_target)

        # oversample
        X_sm, y_sm = prep.resampling(oversample_ratio=oversample_size,
                                     minority_num=train_df['Classification'].value_counts()[1],
                                     majority_num=train_df['Classification'].value_counts()[0],
                                     minority_label=train_df['Classification'].value_counts().index[1],
                                     majority_label=train_df['Classification'].value_counts().index[0])

        # use all data to do training in practice
        X_train_final = X_sm.toarray()
        y_train_final = y_sm


        #array to df
        save_sm_df = aug.ndarray_to_df(x_ndarray = X_train_final,x_columns=save_vectorizer.get_feature_names(),
                                       x_index=len(X_train_final),y_ndarray=y_train_final,y_columns=['Classification'],y_index=len(y_train_final))


        file_name = 'trainingSmote'
        save_sm_df.to_csv(tempFolder+file_name+'.csv',index=False)

        #save data (after SMOTE) in case later retrain
        db.save_data_gridfs(path = tempFolder,file_name=file_name+'.csv',save_filename=file_name,
                                 content_type=file_name,collection=file_name)




        clf_names = list(model_class.classifiers.keys())
        print('Pre-determined models:',clf_names)

        best_model = best_model_name
        clf_model, clf_params = model_class.build_clf(best_model)

        if len(clf_names) ==1:
            #global save_model
            save_model = clf_model.set_params(**clf_params)
            save_model.fit(X_train_final, y_train_final)

            # save trained model to db

            db.compile_model_and_save(
                                      vectorizer = save_vectorizer ,
                                      vectorizer_name = 'TfidfVectorizer',
                                      model=save_model,
                                      model_name = best_model,
                                      best_params = [clf_params] ,
                                      collection = model_collection_name,
                                      path = complied_model)



            print('Saved the best model to db.')
        else:
            print('Please only select the best model ')



    return if_exist,save_model,save_vectorizer




if __name__=='__main__':

    x = pretrain(db_client = 'localhost:27017/', db_name= 'GSA_EULA')






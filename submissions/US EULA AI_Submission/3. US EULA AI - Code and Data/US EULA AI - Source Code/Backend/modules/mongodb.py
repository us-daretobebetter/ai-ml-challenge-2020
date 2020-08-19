import pickle
import pymongo
import datetime
import pandas as pd
from gridfs import GridFS
import os


class save_read_db():

    def __init__(self, db,project_id =None,client='mongodb://localhost:27017/') :
        self.project_id = project_id
        self.client = client
        self.db = db

    def drop_db(self,db):
        myclient = pymongo.MongoClient(self.client)
        print('Initialize database!')
        myclient.drop_database(db)
        return


    def raw_data_to_db(self, dataframe, collection):
        """
        This function is to save pure raw data into database;
        :param dataframe:
        :param collection:
        :return:
        """
        # connect to mongodb
        myclient = pymongo.MongoClient ( self.client )
        mydb = myclient [ self.db ]
        mycol = mydb [ collection ]
        # delete old data
        mycol.delete_many ( {} )

        dataframe.columns = dataframe.columns.str.replace ( ".", "dot" )
        print(dataframe.columns)

        dataframe_to_dict = dataframe.to_dict ( 'records' )

        # insert new data
        mycol.insert_many (dataframe_to_dict )
        print('Save Data into Database Completed')
        return



    def save_traindata_to_db(self, scaler, train_columns, labels, collection='trainingData') :
        # connect to mongodb
        myclient = pymongo.MongoClient ( self.client )
        mydb = myclient [ self.db ]


        train_labels = pickle.dumps ( labels )
        pickled_vec = pickle.dumps (scaler)

        # create 'trainData' colletion
        mycol = mydb [ collection ]
        mycol.insert_one ( {'projectId' : self.project_id,
                            # 'category' : level_proj,
                            'scaler' : pickled_vec,
                            'trainFeatures' : train_columns,
                            'trainLabels' : train_labels,
                            'createdTime' : datetime.datetime.utcnow ()} )
        print ( 'Save training data to db successfully' )
        return

    def save_data_gridfs(self,path,file_name,save_filename,content_type,collection='trainingSmote'):
        """

        :param data:
        :param collection:
        :return:
        """
        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]

        fs = GridFS(mydb,collection)

        with open(path+file_name,'rb') as file:
            fs.put(file,content_type=content_type,filename=save_filename)



        print('----------Saved data using GridFS----------')
        return





    def save_model_to_db(self,
                         vectorizer,
                         #train_data,
                         model, model_name, feature_include, feature_exclude,
                         performance,
                         best_params,
                         #if_retrain,
                         collection='models') :
        myclient = pymongo.MongoClient ( self.client )
        mydb = myclient [ self.db ]
        mycol = mydb [ collection ]

        # pickling the model
        pickled_model = pickle.dumps ( model )
        pickled_vectorizer = pickle.dumps ( vectorizer )
        # saving model to mongodb



        mycol.insert_one ( {
                            'projectId' : self.project_id,
                            'model' : pickled_model,
                            'name' : model_name,
                           'vectorizer' : pickled_vectorizer,
                            #'training_dataset':train_data,
                            'featureInclude' : feature_include,
                            'featureExclude' : feature_exclude,
                            'performance' : performance,
                            'best_params': best_params,
                            #'if_retrain' : if_retrain,
                            'createdTime' : datetime.datetime.utcnow ()} )
        print ( 'Successfully save model to db' )
        return


    def compile_model_and_save(self,vectorizer,vectorizer_name,model,model_name,best_params,collection,path):
        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]
        mycol = mydb[collection]


        # Save the model to local
        model_file = path+model_name+'.pkl'
        vec_file = path+vectorizer_name+'.pkl'
        output_model = open(model_file, 'wb')
        output_vec = open(vec_file, 'wb')
        pickled_model = pickle.dump(model, output_model)
        pickled_vectorizer = pickle.dump(vectorizer,output_vec)
        output_model.close()
        output_vec.close()

        # Put the model to DB
        fs = GridFS(mydb,collection)

        with open(model_file, 'rb') as best_model:
             model1= fs.put(best_model, filename=model_name)

        with open(vec_file, 'rb') as vec:
            vec1 = fs.put(vec, filename=vectorizer_name)


        mycol.update_one({'model_name': model_name},
                         {'$set': {'model': model1,
                                   'vectorizer':vec1,
                                   #'performance':performance,
                                     'best_params':best_params,
                                   'created_time': datetime.datetime.now()}},
                         upsert=True)




        return





    def read_raw_data_from_db(self, collection) :
        myclient = pymongo.MongoClient ( self.client )
        mydb = myclient [ self.db ]
        mycol = mydb [ collection ]

        df = pd.DataFrame ( list ( mycol.find () ) )
        df.columns = df.columns.str.replace ( "dot", "." )

        return df

    def load_saved_data_from_db(self, collection) :
        print ( 'Reading', collection, 'data from mongodb' )
        myclient = pymongo.MongoClient ( self.client )
        mydb = myclient [ self.db ]
        mycol = mydb [ collection ]


        json_data = {}
        data = mycol.find ({})

        for i in data :
            #print(i)

            json_data = i


        # fetching model for db

        return json_data

    def read_gridfs_data(self,file_name,collection,if_df='yes'):

        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]

        #print('---------Read gridfs data from db---------')
        fs=GridFS(mydb,collection=collection) #disable_md5=True
        read = fs.get_last_version(file_name).read()

        if if_df =='yes':

            read = pd.read_csv(read)
            print('---------Load gridfs dataframe from db---------')
        else:
            read = pickle.loads(read)


        return read


    def check_if_exists(self,collection_name,model_path,vec_path):

        myClient = pymongo.MongoClient( self.client )
        mydb = myClient[self.db]

        #check if collection exists

        collections = mydb.list_collection_names()
        model_exists = os.path.exists(model_path)
        vec_exists = os.path.exists(vec_path)
        if model_exists==True and vec_exists==True:
            models_exist = True
        else:
            models_exist = False

        if collection_name not in collections or models_exist==False:

            print('---------no collection:',collection_name,' found---------')

            ifexisit = 0

        else:
            #check if documents exist

            coll_count = mydb[collection_name].count
            print('db collections: ', collections)
            if coll_count == 0:
                print('---------Collection is empty; Will drop it---------')
                mydb[collection_name].drop()
                ifexisit=0
            else:
                ifexisit=1
        return ifexisit


    def update_document(self,object_id,replacement,collection):
        """

        :param object_id:  ObjectId("5cfa4c0ad9edff487939dda5")
        :param replacement:replacement_data = {"some text" : "some text"}
        :param collection:
        :return:
        """

        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]
        col = mydb[collection]

        query = {"_id": object_id}

        # pass the `query` and `replacement_data` objects to the method call
        result = col.update_one(query, {"$set":replacement})

        print('Successfully updated documents with modified prediction results.')

        # json_data = {}
        # data = col.find({}) #load all predictions including modified results
        # for i in data:
        #     json_data = i
        # # fetching model for db

        return

    def load_json_from_db(self, collection,subtitle_col,remove_cols,docid):
        print('Reading', collection, 'data from mongodb')
        myclient = pymongo.MongoClient(self.client)
        mydb = myclient[self.db]
        mycol = mydb[collection]

        return_list=[]
        query = {subtitle_col: ""}

        for col in remove_cols:

            mycol.update_many(query, { '$unset': { col : 1} })

        print('Successfully separate subtitles with clauses')
        #data = mycol.find(query)
        #print(data)
        data = mycol.find({'Doc_Id': docid})
        for i in data:
            # print(i)

            json_data = i
            json_data['_id'] = str(json_data['_id'])
            return_list.append(json_data)
        # fetching model for db

        return return_list




if __name__=='__main__':

    x = save_read_db(db = None, project_id =None,client='mongodb://localhost:27017/')
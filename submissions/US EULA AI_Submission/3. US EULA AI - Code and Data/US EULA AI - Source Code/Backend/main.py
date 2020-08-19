import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import src.nlpFunction as nlpFunction
import src.outputRawCsv as outputRawCsv
import json
from flask import Flask, request, flash
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
from bson.objectid import ObjectId
import string
import random


from modules import mongodb,nlp
import pandas as pd
import pretrained as pre
from sklearn.metrics import brier_score_loss


app = Flask(__name__)
cors = CORS(app)

# All the files is stored
tempFolder = 'temp_file/'

# This is the sub folder for each type of files
# All the files will be uploaded in this folder
UPLOAD_FOLDER = os.path.join(tempFolder + 'uploads')
# Split based re and save the raw split csv in the folders
splittedPdfFolder = os.path.join(tempFolder + 'readFromPDF')
splittedDocFolder = os.path.join(tempFolder + 'readFromDOC')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'docx', 'pdf'}


# TODO: Settings for training/saving model
db_client = 'mongodb://localhost:27017/'
db_name = 'GSA_EULA'
db =mongodb.save_read_db(client=db_client, db=db_name)
target_name = 'Classification'




def get_response(data):

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/getEula', methods=['POST'])
def get_eula():
    print('Upload file route is being called')
    if 'file' not in request.files:
        return get_response('No file part')
    files = request.files.getlist("file")

    global cleanDF
    cleanDF = pd.DataFrame()

    # Create a dic to store the doc id as key and file name as value
    global fileNameDic
    fileNameDic = {}

    for file in files:

        if file.filename == '':
            return get_response('No file selected for uploading')
        # saving the files in upload folder
        if file and allowed_file(file.filename):
            doc_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
            filename = secure_filename(file.filename)
            global name_file
            name_file = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Get the success for frontend
            get_response('success')
        else:
            return get_response('Allowed file types are docx and pdf')

        if name_file.split('.')[-1] == 'pdf':
            # Read pdf file from uploads folder in tempFolder
            df = outputRawCsv.get_Pdf(fileName=name_file, uploadsFolder=UPLOAD_FOLDER, PDFfolder=splittedPdfFolder,
                                      saveToLocal=False)
            processedDF = nlpFunction.NLPcleanTextPDF(df, tempFolder=tempFolder,
                                                  outputFileName=name_file.replace('pdf', 'csv'),
                                                  TYPEdf=True)
            target_file_name = name_file.replace('pdf', 'csv')
        elif name_file.split('.')[-1] == 'docx':
            # Read docx file from uploads folder in tempFolder
            df = outputRawCsv.get_Doc(fileName=name_file, uploadsFolder=UPLOAD_FOLDER, DOCfolder=splittedDocFolder,
                                      saveToLocal=True)
            processedDF = nlpFunction.NLPcleanTextDOC(df, tempFolder=tempFolder,
                                                  outputFileName=name_file.replace('docx', 'csv'),
                                                  TYPEdf=True)
            target_file_name = name_file.replace('docx', 'csv')
        doc_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
        processedDF['Doc_id'] = doc_id
        fileNameDic[doc_id] = filename
        cleanDF = cleanDF.append(processedDF, ignore_index=True)

    print('EULA received!')

    print('Prediction Starts.')
    testing_df = cleanDF
    testing_df = nlp.nlp_cleaning_pre(testing_df, colname='PRE_CLEAN_TEXT', textcol='Clause Text')

    testing_features = save_vectorizer.transform(testing_df['PRE_CLEAN_TEXT'])
    testing_pred_class = save_model.predict(testing_features)
    testing_pred_prob = save_model.predict_proba(testing_features)[:, 0]

    test_pred_df = testing_df.copy().reset_index(drop=True)
    test_pred_df['Predicted_Label'] = pd.Series(testing_pred_class)
    test_pred_df['Acceptance_Proba'] = pd.Series(testing_pred_prob).round(4)
    test_pred_df['Prediction_Confidence_Score'] = test_pred_df['Acceptance_Proba']

    for i in range(test_pred_df.shape[0]):
        try:
            if test_pred_df.loc[i, 'Predicted_Label'] == test_pred_df['Predicted_Label'].value_counts().sort_index().index[1]:
                test_pred_df.loc[i, 'Prediction_Confidence_Score'] = round((1 - test_pred_df.loc[i, 'Acceptance_Proba']), 4)
        except:
            pass

    test_pred_df.rename(columns={'PRE_CLEAN_TEXT': 'Pre_Clean_Text',
                                 'Clause Text': 'Clause_Text',
                                 'Clause ID': 'Clause_Id',
                                 'Full Clause': 'Full_Clause',
                                 'Doc_id': 'Doc_Id'}, inplace=True)


    db.raw_data_to_db(test_pred_df, collection='predict_results')


    return get_response(fileNameDic)




@app.route('/getResult', methods=['POST'])
def get_prediction():
    docid = request.get_json(force=True)['Doc_Id']
    # separate subtitles from clauses by removing predictions of subtitles
    remove_cols = ["Predicted_Label","Acceptance_Proba","Prediction_Confidence_Score"]
    predict_json = db.load_json_from_db(collection='predict_results',subtitle_col ="Clause_Text",
                                        remove_cols = remove_cols, docid=docid)

    return get_response(predict_json)



@app.route('/evaluateClause', methods=['POST'])
def evaluate():
    modify_list = request.get_json(force=True)['modifications']
    print(modify_list)
    print('modify result route')
    save_cols = ['Clause_Id', 'Clause_Text', 'Predicted_Label']

    modify_df = pd.DataFrame()
    for dict1 in modify_list:
        key, value = list(dict1.items())[0]
        obj_id = ObjectId(str(key))
        modify_label = str(float(value))

        replacement_data =  {"Predicted_Label" : modify_label}

        db.update_document(object_id=obj_id,replacement =replacement_data,collection='predict_results')

    save_df = db.read_raw_data_from_db(collection ='predict_results' )
    print('Total Prediction after modification:',save_df.shape[0])
    modify_df = pd.concat([modify_df, save_df], axis=0)

    #print(modify_df)
    modify_df.rename(columns={"Predicted_Label":target_name},inplace=True)


    print('done saving')
    if_retrain = request.get_json(force=True)['retrain']
    if if_retrain==True:
        if_exist1,save_model1,save_vectorizer1 = pre.pretrain(db_client, db_name,if_retrain = 'yes', combined_df = modify_df)

        print('Updated models')
    if if_retrain == False:
        pass

    return get_response('Success!')








if __name__ == '__main__' :

    global if_exist,save_model,save_vectorizer
    if_exist,save_model,save_vectorizer = pre.pretrain(db_client, db_name)
    app.run(debug=True, port=3002)




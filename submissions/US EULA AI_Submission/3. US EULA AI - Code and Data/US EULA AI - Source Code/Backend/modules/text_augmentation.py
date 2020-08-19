import pandas as pd
from nltk.corpus import wordnet
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from collections import Counter
from modules import nlp


#common_cols = ['Clause ID', 'Classification', 'PRE_CLEAN_TEXT']
#upload_data = 'data/'

def get_augmented_df(base_df,common_cols,number_of_synonym = 3):

    def find_synonyms(word):

        synonyms=[]

        for synset in wordnet.synsets(word):
            for syn in synset.lemma_names():
                synonyms.append(syn)

        #drop duplicates adding afterwards but still keep original order(sense importance)
        synonyms_no_duplicates = list(OrderedDict.fromkeys(synonyms))

        return synonyms_no_duplicates


    def generate_new_sentence(sentence,max_replaced_words=3):
        new_sentences = []
        sentence = str(sentence).replace(r'[\d]+','')
        for word in word_tokenize(sentence):
            replace_words = find_synonyms(word)[0:max_replaced_words]

            if len(word)<=3:
                # ignore word with length less than 3
                continue
            for rep in replace_words:
                # some synonyms may have _ ; replace it with whitespace
                rep = rep.replace('_',' ')
                new_line = sentence.replace(word,rep)
                new_sentences.append(new_line)

            return new_sentences


    def augment_text_df(df,text_col,label_col,max_replaced_words=number_of_synonym,id_col ='Clause ID',aug_text_col='AUG_CLEAN_TEXT'):
        new_df=pd.DataFrame([],columns = [aug_text_col,id_col,label_col])
        print('shape of the base dataframe to be augmented: ',df.shape)
        for idx,text in enumerate(df[text_col]):

            inner_df=pd.DataFrame()
            new_sentences = generate_new_sentence(text,max_replaced_words=max_replaced_words)

            inner_df[aug_text_col] = pd.Series(new_sentences)
            inner_df[id_col]= df.loc[idx,id_col]
            inner_df[label_col] = df.loc[idx,label_col]

            new_df = pd.concat([new_df,inner_df]).reset_index(drop=True)

        print('Finish text augmentation: augmented df shape: ',new_df.shape)

        return new_df




    base_df1 = base_df.reset_index(drop=True)
    base_df1 = nlp.nlp_cleaning_pre(base_df1, colname='PRE_CLEAN_TEXT', textcol='Clause Text')

    # drop nan values
    base_df1.drop(base_df1[base_df1['Classification'] == 'nan'].index, inplace=True)
    base_df1.drop(base_df1[(base_df1['Classification'] == 'nan') | (base_df1.isna().any(axis=1))].index, inplace=True)
    #generate augmented text for unacceptable clauses in base_df
    new_df = base_df1[base_df1['Classification']==base_df1['Classification'].value_counts().sort_index().index[1]].reset_index(drop=True)

    new_df = augment_text_df(new_df, text_col='PRE_CLEAN_TEXT', label_col='Classification', max_replaced_words=number_of_synonym,
                             id_col='Clause ID', aug_text_col='AUG_CLEAN_TEXT')
    print('augmented_unaccepted_clauses',new_df.shape)
    new_df['Classification'] = new_df['Classification'].astype(float).astype(str)

    new_df = nlp.nlp_cleaning_pre(new_df, colname='PRE_CLEAN_TEXT', textcol='AUG_CLEAN_TEXT')


    aug_df = new_df[common_cols]
    train_df = base_df1[common_cols]

    combined_df = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    combined_df['Classification'] = combined_df['Classification'].astype(float).astype(str)

    combined_df = combined_df.dropna().reset_index(drop=True)
    combined_df.drop_duplicates(subset=['PRE_CLEAN_TEXT'], inplace=True)

    #combined_df.columns
    #combined_df.shape
    Counter(combined_df['Classification'])
    #combined_df.isna().sum()

    #drop rows with empty text cell or null values
    combined_df.drop(combined_df[(combined_df['Classification'] == 'nan')|(combined_df.isna().any(axis=1))].index, inplace=True)
    combined_df['Classification'] = combined_df['Classification'].astype(float).astype(str)


    return combined_df,aug_df








def ndarray_to_df(x_ndarray, x_columns, x_index, y_ndarray, y_columns, y_index):
    """

    :param ndarray:
    :param columns:
    :param index:
    :return:
    """
    x_index = [i for i in range(x_index)]
    y_index = [i for i in range(y_index)]

    x_df = pd.DataFrame(x_ndarray, columns=x_columns, index=x_index)
    y_df = pd.DataFrame(y_ndarray, columns=y_columns, index=y_index)

    df = pd.concat([x_df, y_df], axis=1)

    return df



#TODO: remove backups
#upload_data = 'data/'
# #train_df = pd.read_csv(upload_data+'updated_nlp_cleaning_data.csv')
# #train_df.shape
# #train_df.columns
# train_df = train_df.reset_index(drop=True)
# new_df = augment_text_df(train_df,text_col='PRE_CLEAN_TEXT',label_col='Classification',max_replaced_words=3,id_col ='Clause ID',aug_text_col='AUG_CLEAN_TEXT')

# new_df.shape
# Counter(new_df['Classification'])
# #new_df.to_csv(upload_data+'augmented_data_3sym_80pc.csv',index=False)
# new_df = pd.read_csv(upload_data+'augmented_data_3sym.csv')
#
# #process new training the same way
# db_client = 'mongodb://admin:innovation@localhost:27017/'
# db_name = 'GSA_EULA'
# db =mongodb.save_read_db(client=db_client, db=db_name)
# #transform type
# new_df['Classification'] = new_df['Classification'].astype(float).astype(str)
# new_df.columns
# new_df = nlp.nlp_cleaning_pre(new_df, colname='PRE_CLEAN_TEXT', textcol='AUG_CLEAN_TEXT')
#
#
# #combine two dfs
# common_cols = ['Clause ID', 'Classification', 'PRE_CLEAN_TEXT']
# aug_df = new_df[common_cols]
# train_df = train_df[common_cols]
# combined_df = pd.concat([train_df,aug_df],axis=0).reset_index(drop=True)
# combined_df['Classification'] = combined_df['Classification'].astype(float).astype(str)
#
# combined_df = combined_df.dropna().reset_index(drop=True)
# combined_df.drop_duplicates(subset=['PRE_CLEAN_TEXT'],inplace=True)
#
# combined_df.columns
# combined_df.shape
# Counter(combined_df['Classification'])
# combined_df.isna().sum()
#
# #combined_df.to_csv(upload_data+'80pc_train_and_augmented_data_updated_nlp.csv',index=False)
# combined_df = pd.read_csv(upload_data+'80pc_train_and_augmented_data_updated_nlp.csv')



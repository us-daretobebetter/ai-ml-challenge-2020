import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

##
# cleaning process
def nlp_cleaning_pre(df, colname='PRE_CLEAN_TEXT', textcol='ORIGINAL_ITEM_DESCRIPTION'):
    """
    steps:
           all lowercase,
           split lines, replace \r\n\t
           save all digits except item number : match digit(dot)digit like 2.7 or 1.2. or 12.
           punctuations: save , . $ = @ * -
                        remove () | / : ; # ? ! & " '
           remove useless patterns, defined by txt file, update if needed

    parameter: ## path, if put in same directory, put file name --> move to nlp_post
    note: plz perform both on training, validation and testing
    """

    df_p = df.copy()

    df_p[colname] = df_p[textcol].astype(str).str.lower()

    df_p[colname] = df_p[colname].replace({'\r\n': ' ', '\t': ' '}, regex=True)

    # remove igf::ct::igf pattern
    #df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'[a-z]{1,3}[:]{1,2}[a-z]{1,3}[:]{1,2}[a-z]{1,3}', ' ', x))

    #remove item number : match digit(dot)digit like 2.7 or 1.2. or 12.
    pat = r'[\d]+[.]+[\d]?.?'
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(pat,'',x))

    #remove the combination of underlines and digits
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'\d*_+\d*', '', x))
    print('Updated NLP!')
    # remove punctuations
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'[^\w|^\s]+', '', x))
    # TODO: remove additional punctuations
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'[“’‘”]', '', x))

    # remove rest punctuations
    #df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'[,.$=@*]+', ' ', x))  # preserve -

    #remove single characters
    #df_p[textcol] = df_p[textcol].apply(lambda x: re.sub(r'\s+[a-z]{1}\s+', ' ', x))

    # remove percentage \d+[\s]?%
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'\s+\d{1,3}\s?(?:\.\s?\d{0,2})?[%]', ' ', x))



    # stop words, stem
    stop_words = set(stopwords.words('english'))
    #
    # porter = PorterStemmer()
    # df_p[colname] = df_p[colname].apply(lambda x: ' '.join([porter.stem(word) for word in x.split() if word not in stop_words]))

    stemmer = WordNetLemmatizer()
    df_p[colname] = df_p[colname].apply(
        lambda x: ' '.join([stemmer.lemmatize(word) for word in x.split() if word not in stop_words]))


    # remove duplicated whitespace
    df_p[colname] = df_p[colname].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()


    return df_p


def vectorize(df,colname,max_features_num=5000):

    tfidf = TfidfVectorizer(max_features=max_features_num)
    tfidf = tfidf.fit(df[colname])
    tfidf_trans = tfidf.transform(df[colname])

    return tfidf_trans


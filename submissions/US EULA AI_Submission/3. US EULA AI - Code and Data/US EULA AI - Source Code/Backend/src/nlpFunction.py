import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords

tempFolder = '../temp_file/'



def checkLengthAndEnd(line, lenLimit=50):
    # Filter out the line with sting 'COMPANY INC.'
    if line.strip() == 'COMPANY INC.':
        return ''

    if len(line) < lenLimit and line.strip() != '':
        if line.strip()[-1] not in ['.', ';', ',']:
            # Consider the condition when phrases are upper case with a colon
            if line.isupper() == True and line.strip()[-1] == ':':
                return line
            # Capture patterns like "DEFINITIONS:  As used in this agreement:"
            if ':' in line:
                if line.split(':')[0].isupper():
                    return line
            # Capture patterns like "hereby agree as follows:"
            if line.islower():
                return line
            else:
                # print(len(line), line)
                return ''
        else:

            return line
    else:

        return line


# This is for pdf use now
def NLPcleanTextPDF(file, tempFolder, outputFileName=None, TYPEdf = False, TYPEcsv = False):
    if TYPEdf == True:
        df = file
    elif TYPEcsv == True:
        df = pd.read_csv(os.path.join(tempFolder, 'readFromPDF/', file))

    ID = []
    TEXT = []
    fullClause = []

    for index, row in df.iterrows():
        id = row['Clause ID'],
        text = row['Clause Text']
        item = row['reTitle']
        clause = ''
        for line in text.split('\n'):
            line = checkLengthAndEnd(line)
            if line not in ['', None]:
                clause = ''.join((clause, line))

        clause = "".join([s for s in clause.splitlines(True) if s.strip()])
        # Replace the double quote sign
        clause = clause.replace('“', '"').replace('”', '"')

        itemNumWithClause = str(item) + ' ' + clause

        # Get the string that stopwords were removed
        stop_words = stopwords.words('english')
        # Remove the stopwords in order to check the subtitle
        nonStopwordsText = ' '.join([word for word in clause.split() if word.lower() not in stop_words])


        # Check if the string contains item number, also use istitle() and isupper() to check the string
        if clause != '' and len(re.findall(r'(^[a-zA-Z0-9_]{1,2}[.])', itemNumWithClause)) == 1 and (
                nonStopwordsText.istitle() or nonStopwordsText.isupper()) and len(clause) < 60:
            ID.append(id[0])
            TEXT.append('')
            fullClause.append(str(item) + ' ' + clause)
        elif clause != '':
            ID.append(id[0])
            TEXT.append(clause)
            fullClause.append(str(item) + ' ' + clause)


    Newdf = pd.DataFrame(list(zip(ID, TEXT, fullClause)), columns =['Clause ID', 'Clause Text', 'Full Clause'])
    Newdf.to_csv(os.path.join(tempFolder, 'outputFile/', outputFileName), index=False)
    return Newdf

# NLPcleanTextPDF(file='sample_eula_1.csv', outputFileName='sample_eula_1PDF.csv', tempFolder=tempFolder, TYPEcsv=True)


def NLPcleanTextDOC(file, tempFolder, outputFileName=None, TYPEdf = False, TYPEcsv = False):
    global lowerEndClause
    lowerEndClause = None
    if TYPEdf == True:
        df = file
    elif TYPEcsv == True:
        df = pd.read_csv(os.path.join(tempFolder, 'readFromDOC/', file))

    ID = []
    TEXT = []
    fullClause = []

    for index, row in df.iterrows():
        id = row['Clause ID'],
        text = row['Clause Text']
        clause = text.strip()

        # Get the clause without the stop words
        stop_words = stopwords.words('english')
        # Remove the stopwords in order to check the subtitle
        nonStopwordsText = ' '.join([word for word in clause.split() if word.lower() not in stop_words])

        # Set multiple conditions to check if the string is subtitle or clauses
        if ((clause != '' and len(clause) > 20 and clause.strip()[-1] in ['.', ';', ',']) or len(clause) > 100) and (
            nonStopwordsText.istitle() == False) and len(clause) != 1 and (clause.isupper() == False or (clause.isupper(
                                        ) == True and len(clause.strip()) > 100)):
            ID.append(id[0])
            TEXT.append(clause)
            fullClause.append(clause)

        else:
            ID.append(id[0])
            TEXT.append('')
            fullClause.append(clause)

    Newdf = pd.DataFrame(list(zip(ID, TEXT, fullClause)), columns =['Clause ID', 'Clause Text', 'Full Clause'])
    Newdf.to_csv(os.path.join(tempFolder, 'outputFile/', outputFileName), index=False)
    return Newdf

# NLPcleanTextDOC(file='sample_eula_1.csv', outputFileName='sample_eula_1DOC.csv', tempFolder=tempFolder, TYPEcsv=True)



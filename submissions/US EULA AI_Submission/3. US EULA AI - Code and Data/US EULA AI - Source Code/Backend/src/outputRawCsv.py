import pandas as pd
import fitz
import re
import os
import docx

# Parse PDF files into string, use item number as delimiters
def get_Pdf(fileName, uploadsFolder, PDFfolder, saveToLocal = False):

    dfDic = {}
    ID = 1000
    pdf_file = fitz.open(os.path.join(uploadsFolder, fileName))
    fullPage = ''
    for i in range(pdf_file.pageCount):
        page = pdf_file.loadPage(i)
        # Replace messy code to double quote sign
        text = page.getText().replace('“', '"').replace('”', '"')
        text = "".join([s for s in text.splitlines(True) if s.strip()])
        fullPage = ''.join((fullPage, text))

    # Use re to find pattern like "a.", "1.2" or "i."
    pattern = '\n[a-zA-Z][.]\s|\n[0-9_]{1,3}[.]\d{0,2}[.]?\d{0,2}\s|\n[x]?[i]{0,3}[x]?[v]?[i]{0,3}[.]\s|\n[(]\w[)]\s'
    match = re.findall(pattern, fullPage)
    match = ['\n'] + match
    for i in re.split('\n[a-zA-Z][.]\s|\n[0-9_]{1,3}[.]\d{0,2}[.]?\d{0,2}\s|\n[x]?[i]{0,3}[x]?[v]?[i]{0,3}[.]\s|\n[(]\w[)]\s', fullPage):
        dfDic[ID] = i.strip()
        ID += 1
    df = pd.DataFrame.from_dict(dfDic, orient='index')
    df = df.reset_index().rename(columns={0: 'Clause Text', 'index': 'Clause ID'})
    df['reTitle'] = [str(i.strip()) for i in match]

    if saveToLocal == True:
        df.to_csv(os.path.join(PDFfolder, fileName.replace('pdf', '') + 'csv'), index=False)
    return df

# uncomment to test the local function
# get_Pdf(fileName='sample_eula_1.pdf',tempFolder=tempFolder, saveToLocal = True)


# Parse MS Word document into clauses/paragraphs, the docx package can parse document into paragraphs
def get_Doc(fileName, uploadsFolder, DOCfolder, saveToLocal = False):
    dfDic = {}
    ID = 1000
    doc = docx.Document(os.path.join(uploadsFolder, fileName))
    for i in doc.paragraphs:
        for para in i.text.split('\n'):
            para = para.strip().replace('“', '"').replace('”', '"').replace('–', '-').replace('’', "'")

            if para.strip() != '':
                if para[0].islower():
                    ID = ID - 1
                    dfDic[ID] = dfDic[ID] + ' ' + para
                    ID += 1
                else:
                    dfDic[ID] = para
                    ID += 1

    df = pd.DataFrame.from_dict(dfDic, orient='index')
    df = df.reset_index().rename(columns={0: 'Clause Text', 'index': 'Clause ID'})

    if saveToLocal == True:
        df.to_csv(os.path.join(DOCfolder, fileName.replace('docx', 'csv')), index=False)
    return df

# uncomment to test the local function
# get_Doc('sample_eula_1.docx',uploadsFolder=UPLOAD_FOLDER, DOCfolder=splittedDocFolder, saveToLocal = True)

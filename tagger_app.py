#!/usr/bin/python3.6

# Machine Learning Question Tagger from StackOverFlow

'''
This Flask web app extract keywords from user text input
A supervised model is used to find tags.
If this model doesn't find, a unsupervised model outputs 4 Tags.

Models were trained and optimized over 17000 questions.

Supervised model : RandomForestClassifier
Unsupervied model : LatentDirichletAllocation + matrix Distrib tags = f(topics)

'''

from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
import sklearn
from sklearn.externals import joblib
import string
from string import punctuation
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.snowball import EnglishStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
# import user module
from my_text_utils import myTokenizer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    ### definitions : ###
    # model
    mdlFileName = 'mdl_cmp_RF_tags51_max_depthNone_max_features31_min_samples_split2_n_estimators25.pkl'
    # stop words
    stopWordsFileName = 'stop_words_sw.pkl'
    # count vectorizer
    countVectFileName = 'cvect_tags51.pkl'
    # tfidf
    tfidfFileName = 'tfidf_tags51.pkl'
    # MultiLabelBinarizer
    mlbFileName = 'mlb_tags51.pkl'
    # df Topics tags
    dfTopicsTagsFileName = 'df_topics_tags_top100.pkl'
    # mode unsup LDA
    mdlUnsupFileName = 'model_LDA__learning_decay0.7_max_iter20_n_components100.pkl'
    # count vectorizer unsup
    countVectUnsupFileName = "cvect_lda.pkl"

    if (request.method == 'POST'):
        message = request.form['message']
        #data = [message]
        #vect = cv.transform(data).toarray()
        #my_prediction = clf.predict(vect)

        # prepare dictionnary of translation to suppress ponctuation
        replace_punctuation = str.maketrans(string.punctuation,
            ' '*len(string.punctuation))
        # load stopwords
        sw = joblib.load(stopWordsFileName)
        # clean text
        message = cleaning_text(message, sw, replace_punctuation)
        #load CounterVectorizer
        tf_vectorizer_sup_1 = joblib.load(countVectFileName)
        # countvectorizing
        contVectValue = tf_vectorizer_sup_1.transform([message])
        # load  TfidfTransformer
        tfidf_transformer_sup_1 = joblib.load(tfidfFileName)
        # tfidf
        tfIdfValue = tfidf_transformer_sup_1.transform(contVectValue)
        # load model
        myModel = open(mdlFileName, 'rb')
        clf = joblib.load(myModel)
        # predict
        encoded_y_pred = clf.predict(tfIdfValue)
        # load MultiLabelBinarizer
        mlb = joblib.load(mlbFileName)
        # decode tags
        tu_tags = mlb.inverse_transform(encoded_y_pred)
        # check tags found ?
        flag_tag_found = len(tu_tags[0]) > 0
        # if not, tray to predict with Unsupervised model
        if flag_tag_found == False:
            # load
            df_topics_tags = joblib.load(dfTopicsTagsFileName)
            # load model
            myModelUnsup = open(mdlUnsupFileName, 'rb')
            model_lda = joblib.load(myModelUnsup)
            # load count vect
            tf_vectorizer_1 = joblib.load(countVectUnsupFileName)
            # predict
            tu_tags[0] = find_tags_from_text(text=message,
                tf_vectorizer=tf_vectorizer_1,
                lda_model = model_lda, df_topics_tags=df_topics_tags,
                no_max=4)

        # output
        str_out=""
        for item in tu_tags[0]:
            str_out += item + " "
        message = str_out

    return render_template('result.html', message = message)


if __name__ == '__main__':
	app.run(debug=True)


def cleaning_text(questions_curr, sw, replace_punctuation):
    '''
    Cleaning text before tokenize:
    - lowering,
    - suppress special characters,
    - delete stopwords...
    '''
    # lower case
    questions_curr = ' '.join([w.lower() for w in \
                               nltk.word_tokenize(questions_curr) \
                              if not w.lower() in list(sw)])
    # delete newlines
    questions_curr = re.sub(r'\s+', ' ', questions_curr)
    # delete single quotes
    questions_curr = re.sub(r"\'", " ", questions_curr)
    # delete tags
    questions_curr = re.sub('<[^<]+?>',' ', questions_curr)
    # delete numbers (forming group = word with only numbers
    # example : delete "123" but not "a123")
    questions_curr = re.sub(r'\b\d+\b','', questions_curr)
    # delete ponctuation (replace by space)
    questions_curr = questions_curr.translate(replace_punctuation)

    return questions_curr

def find_tags_from_text(text, tf_vectorizer, lda_model,
                        df_topics_tags, no_max=10):
    '''
    Predict tags from text using tf, lda and tags2topic table

    tf vectorizer, lda and table must be input.
    '''
    # clean the text
    text_cleaned = text

    # calculate feature from text with tf already fitted
    feat_curr =  tf_vectorizer.transform([text_cleaned])

    # calculate topic distrib with lda model already fitted
    topic_distrib_pred = lda_model.transform(feat_curr)

    # find best topic from table df_topics_tags
    return find_tags_from_dtopics(topic_distrib_pred, df_topics_tags,
                                  no_max=no_max)

def find_tags_from_dtopics(d_topics, df_topics_tags, no_max=10):
    '''
    Find best no_max Tags from Topics by giving Topic number as input.
    (By default no_max = 10)
    Uses table linking Tags & Topics

    inputs :
    - d_topics : topics distribution from LDA
    - df_topics_tags : table linking Tags to Topics (By default df_topics_tags)
    - no_max : number of best Tags to output

    returns the list of no_max Topics numbers (int)
    '''

    # multiply topics each columns of df_topics_tags by distrib dtopics vector:
    arr_tags = df_topics_tags.values*d_topics # table (tags(row)*Topics(col))
    # sum each row (by Tags)
    sum_distrib_tags = arr_tags.sum(axis=1) # vector (n Tags)
    # create dataframe to link with tags
    df_sum_tags = pd.DataFrame(data=sum_distrib_tags, columns=["d_sum"],
                           index=df_topics_tags.index)
    # return no_max Tags with best score
    return list(df_sum_tags.sort_values(by="d_sum",
                                        ascending=False).head(no_max).index)


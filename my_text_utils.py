# -*-coding:utf-8 -*

# Module tools for tokenizing step with CountVectorizer
import nltk
from nltk.stem.snowball import EnglishStemmer
import re

def myTokenizer(text):
    '''
    Create tokens from text (English words > 3 letters)
    '''
    def stem_tokens(tokens, stemmer):
        '''
        Stem words in tokens.
        and suppress word < 3 characters
        '''
        stemmed = []
        for item in tokens:
            if re.match('[a-zA-Z0-9]{3,}',item):
                stemmed.append(stemmer.stem(item))
        return stemmed

    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, EnglishStemmer())
    return stems

# test de la fonction myTokenizer
if __name__ == "__main__":
    print(myTokenizer("e 123 e the module is firebase"))
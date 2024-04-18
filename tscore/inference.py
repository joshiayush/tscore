import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import re
import string
import joblib
import pathlib

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from tscore.constants import *

_CLF_PATH = pathlib.Path('.').parent / 'models' / 'tscore.joblib'


class BasePreprocessorMixin(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self


class ExpandContractions(BasePreprocessorMixin):
    _contractions_re = re.compile('(%s)' % '|'.join(CONTRACTIONS.keys()))

    @staticmethod
    def _expand_contractions(text):
        def replace(match):
            return CONTRACTIONS[match.group(0)]

        return ExpandContractions._contractions_re.sub(
          replace, text, re.IGNORECASE
        )

    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self._expand_contractions)
        return X


class TextCleaner(BasePreprocessorMixin):
    @staticmethod
    def _clean_punctuations(text):
        translator = str.maketrans(
          string.punctuation, ' ' * len(string.punctuation)
        )
        return text.translate(translator)

    @staticmethod
    def _clean_repeating_chars(text):
        return re.sub(r'(.)1+', r'1', text)

    @staticmethod
    def _remove_mentioned(text):
        return re.sub(r'@[A-Za-z0-9]+', '', text)

    @staticmethod
    def _clean_urls(text):
        return re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)

    @staticmethod
    def _clean_numeric_chars(text):
        return re.sub('\w*\d\w*', '', text)

    def _clean_text(self, text):
        text = text.lower()
        text = self._clean_repeating_chars(text)
        text = self._remove_mentioned(text)
        text = self._clean_urls(text)
        text = self._clean_punctuations(text)
        text = self._clean_numeric_chars(text)
        return text

    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self._clean_text)
        return X


class WordTokenizer(BasePreprocessorMixin):
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(word_tokenize)
        return X


class WordLemmatizer(BasePreprocessorMixin):
    @staticmethod
    def _get_word_net_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
          "J": wordnet.ADJ,
          "N": wordnet.NOUN,
          "V": wordnet.VERB,
          "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def _lemmatize_text(tokens):
        lemmatizer = nltk.WordNetLemmatizer()
        return [
          lemmatizer.lemmatize(word, WordLemmatizer._get_word_net_pos(word))
          for word in tokens
        ]

    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self._lemmatize_text)
        return X


class StopWordsRemover(BasePreprocessorMixin):
    @staticmethod
    def _clean_stop_words(tokens):
        return [word for word in tokens if word not in STOP_WORDS]

    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(self._clean_stop_words)
        return X


class JoinTokens(BasePreprocessorMixin):
    def transform(self, X, y=None):
        for column in self.columns:
            X[column] = X[column].apply(lambda tokens: " ".join(tokens))
        return X


pipeline = Pipeline(
  steps=[
    ("expand_contractions", ExpandContractions(columns=["SentimentText"])),
    ("text_cleaner", TextCleaner(columns=["SentimentText"])),
    ("word_tokenizer", WordTokenizer(columns=["SentimentText"])),
    ("word_lemmatizer", WordLemmatizer(columns=["SentimentText"])),
    ("stop_words_remover", StopWordsRemover(columns=["SentimentText"])),
    ("join_tokens", JoinTokens(columns=["SentimentText"])),
  ]
)


def sentiments(texts: list) -> np.array:
    tscore_df = pd.DataFrame({'SentimentText': texts})
    tscore_df_transformed = pipeline.transform(tscore_df.copy())

    X = tscore_df_transformed['SentimentText']
    clf = joblib.load(_CLF_PATH)
    return clf.predict(X), clf.predict_proba(X)

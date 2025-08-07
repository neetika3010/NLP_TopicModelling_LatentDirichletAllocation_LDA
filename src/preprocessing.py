import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(texts):
    """
    Clean, tokenize, remove stopwords and lemmatize input texts.

    Args:
        texts (List[str]): List of raw documents.

    Returns:
        List[List[str]]: List of preprocessed token lists.
    """
    processed = []
    for doc in texts:
        tokens = simple_preprocess(str(doc), deacc=True)  # remove punctuation
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        processed.append(tokens)
    return processed

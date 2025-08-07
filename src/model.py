from gensim import corpora, models

def train_lda_model(tokenized_docs, num_topics=5, passes=10, random_state=42):
    """
    Train an LDA model using Gensim.

    Args:
        tokenized_docs (List[List[str]]): Preprocessed text.
        num_topics (int): Number of topics to extract.
        passes (int): Number of passes through the corpus during training.
        random_state (int): Seed for reproducibility.

    Returns:
        lda_model, corpus, dictionary
    """
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state
    )
    return lda_model, corpus, dictionary

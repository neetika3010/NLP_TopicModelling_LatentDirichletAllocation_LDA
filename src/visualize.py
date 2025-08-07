import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

def visualize_lda_pyldavis(lda_model, corpus, dictionary, output_html_path='outputs/visualizations/lda_vis.html'):
    """
    Save interactive LDA visualization as HTML.

    Args:
        lda_model: Trained LDA model.
        corpus: Corpus used for training.
        dictionary: Gensim dictionary.
        output_html_path (str): Path to save HTML.
    """
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_html_path)
    print(f"LDA visualization saved to {output_html_path}")

def plot_topic_distribution(lda_model, num_words=10):
    """
    Plot keywords per topic using matplotlib.

    Args:
        lda_model: Trained LDA model.
        num_words (int): Number of top words per topic to show.
    """
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    for topic_no, topic in topics:
        words, weights = zip(*topic)
        plt.figure(figsize=(8, 4))
        plt.barh(words, weights)
        plt.title(f"Topic {topic_no}")
        plt.xlabel("Word Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

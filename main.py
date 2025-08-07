# main.py

import os
from src.data_loader import load_data
from src.preprocessing import preprocess_text
from src.model import train_lda_model
from src.visualize import visualize_lda_pyldavis, plot_topic_distribution
from src.utils import save_model

# Set paths
DATA_PATH = 'data/raw/your_data.csv'     # Replace with your actual file name
TEXT_COLUMN = 'text'                     # Replace with your actual text column name
NUM_TOPICS = 5
MODEL_SAVE_PATH = 'outputs/models/lda_model.pkl'
HTML_VIS_PATH = 'outputs/visualizations/lda_vis.html'

def main():
    print("🚀 Loading data...")
    raw_texts = load_data(DATA_PATH, text_column=TEXT_COLUMN)

    print("🧹 Preprocessing text...")
    tokenized_texts = preprocess_text(raw_texts)

    print("Training LDA model...")
    lda_model, corpus, dictionary = train_lda_model(tokenized_texts, num_topics=NUM_TOPICS)

    print("Saving model...")
    save_model(lda_model, MODEL_SAVE_PATH)

    print("Generating visualizations...")
    os.makedirs(os.path.dirname(HTML_VIS_PATH), exist_ok=True)
    visualize_lda_pyldavis(lda_model, corpus, dictionary, output_html_path=HTML_VIS_PATH)
    plot_topic_distribution(lda_model)

    print("Done!")

if __name__ == "__main__":
    main()

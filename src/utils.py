import os
import pickle

def save_model(model, path='outputs/models/lda_model.pkl'):
    """
    Save model to disk using pickle.

    Args:
        model: Gensim model or any object.
        path (str): Output path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path='outputs/models/lda_model.pkl'):
    """
    Load model from disk.

    Args:
        path (str): File path.

    Returns:
        Loaded model.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

from sentence_transformers import SentenceTransformer

_model = None

def get_sentence_embedding(sentence):
    global _model
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = _model.encode(sentence)
    return embeddings

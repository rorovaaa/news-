from sentence_transformers import SentenceTransformer
import numpy as np
import os


class TextEmbedder:
    def __init__(self, model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, text):
        if not text or text.strip() =="":
            return np.zeros(self.model.get_sentence_embedding_dimension())
        embedding = self.model.encode(text)
        return embedding

    def encode_batch(self, texts):
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([])
        embeddings = self.model.encode(valid_texts)
        return embeddings

    def save(self, embeddings, file_path):
        np.save(file_path, embeddings)

    def load_embeddings(self, file_path):
        embeddings = np.load(file_path)
        return embeddings
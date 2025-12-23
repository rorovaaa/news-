import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class SemanticSearcher:
    def __init__(self, embeddings_path, metadata_path):
        self.embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        self.filenames = metadata['filenames']
        self.texts = metadata['texts']
        self.cleaned_texts = metadata['cleaned_texts']
        print(len(self.filenames))
        print(self.embeddings.shape)

    def search(self, query_embedding, top_k=5, threshold=0.3):
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in sorted_indices[:top_k]:
            score = float(similarities[idx])
            
            if score < threshold:
                continue
        
            result = {
                'score': score,
                'filename': self.filenames[idx],
                'text': self.texts[idx][:500] + '...',  
                'index': int(idx),
                'cleaned_text': self.cleaned_texts[idx]
            }
            results.append(result)
        
        return results
    
    def search_by_text(self, query_text, embedder, preprocessor, top_k=5, threshold=0.3):

        processed_query = preprocessor.normalaize(query_text)
        query_embedding = embedder.encode(processed_query)
        return self.search(query_embedding, top_k, threshold)
    
    def print_results(self, results, query=None):
        
        if query:
            print(query)
           
        
        if not results:
            print(" Ничего не найдено")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}.  {result['filename']}")
            print(f"    Схожесть: {result['score']:.3f}")
            print(f"    Текст: {result['text']}")
            print(f"    Индекс: {result['index']}")
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocessor import AdvancedPreprocessor
from src.embedder import TextEmbedder

def process_all_documents():
    preprocessor = AdvancedPreprocessor(language= 'russian', use_lemmatization= True)
    embedder = TextEmbedder()
    input_dir = "data/raw/documents/"
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(1)
    
    texts = []
    cleaned_texts = []
    filenames = []

    for filename in files:
        filepath = os.path.join(input_dir, filename)

        try:
            with open(filepath, 'r', encoding= 'utf-8') as f:
                orig = f.read()

            normalized = preprocessor.normalaize(orig)
            texts.append(orig)
            cleaned_texts.append(normalized)
            filenames.append(filename)

        except Exception as e:
            print(f"не работает  {filename}: {e}")
    print(f"\nУспешно: {len(texts)} фалов ")

    embeddings = embedder.encode_batch(cleaned_texts)
    save_dir = "data/embeddings/"
    os.makedirs(save_dir, exist_ok= True)
    embedder.save(embeddings, os.path.join(save_dir, "embeddings.npy"))

    metadata = {
        'filenames': filenames,
        'texts': texts,
        'cleaned_texts': cleaned_texts,
        'count': len(filenames)
    }
    with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return embeddings, filenames, texts

if __name__ == "__main__":
    process_all_documents()

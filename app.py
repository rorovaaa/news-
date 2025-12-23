import os
import sys
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.preprocessor import AdvancedPreprocessor
    from src.embedder import TextEmbedder
    from src.searcher import SemanticSearcher
except ImportError as e:
    print(f" Ошибка импорта: {e}")
    print("Убедитесь, что файлы в папке src/ существуют")
    sys.exit(1)


class SemanticSearchApp:

    def __init__(self):
        self.preprocessor = None
        self.embedder = None
        self.searcher = None
        self.embeddings_loaded = False
        self.embeddings_path = "data/embeddings/embeddings.npy"
        self.metadata_path = "data/embeddings/metadata.json"
        
        self.documents_path = "data/raw/documents/"
        self.processed_path = "data/embeddings/"
        
        self.stats = {
            'total_searches': 0,
            'average_results': 0,
            'queries_history': []
        }
    
    def initialize(self):
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)

        self.preprocessor = AdvancedPreprocessor(language='russian', use_lemmatization=True)
        self.embedder = TextEmbedder()
        
        if os.path.exists(self.embeddings_path) and os.path.exists(self.metadata_path):
            try:
                self.searcher = SemanticSearcher(self.embeddings_path, self.metadata_path)
                self.embeddings_loaded = True
                print(f" Загружено документов: {len(self.searcher.filenames)}")
            except Exception as e:
                print(f" Ошибка загрузки эмбеддингов: {e}")
                self.embeddings_loaded = False
        else:
            print(" Эмбеддинги не найдены. Сначала обработайте документы.")
            self.embeddings_loaded = False
    
    def process_documents(self):
       
        print("\n ОБРАБОТКА ДОКУМЕНТОВ")
        files = [f for f in os.listdir(self.documents_path) if f.endswith('.txt')]
        if not files:
            print(f" В папке {self.documents_path} нет .txt файлов")
            print("Добавьте текстовые файлы в эту папку и попробуйте снова.")
            return
        
        print(f"Найдено документов: {len(files)}")
        for f in files:
            print(f" {f}")
        
        texts = []
        filenames = []
        original_texts = []
        
        for filename in files:
            filepath = os.path.join(self.documents_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    original = f.read()
                
                normalized = self.preprocessor.normalaize(original)
                
                texts.append(normalized)
                original_texts.append(original)
                filenames.append(filename)
                
                print(f"Обработан: {filename}")
                
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")
        
        if not texts:
            print(" Не удалось обработать ни одного документа, что за хуйню ты принёс")
            return
        try:
            embeddings = self.embedder.encode_batch(texts)
            print(f" Создано эмбеддингов: {len(embeddings)}")
            print(f" Размерность: {embeddings[0].shape}")
        except Exception as e:
            print(f" Ошибка при создании эмбеддингов: {e}")
            return
        print("\n Сохраняю данные...")
        try:
            np.save(self.embeddings_path, embeddings)
            metadata = {
                'filenames': filenames,
                'texts': original_texts,
                'cleaned_texts': texts,
                'created_at': datetime.now().isoformat(),
                'embedding_dimension': embeddings[0].shape[0],
                'total_documents': len(filenames)
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.searcher = SemanticSearcher(self.embeddings_path, self.metadata_path)
            self.embeddings_loaded = True
            
            print(f" Успешно сохранено!")
            print(f" Документов: {len(filenames)}")
            print(f" Размерность эмбеддингов: {embeddings[0].shape}")
            print(f" Путь: {self.embeddings_path}")
            
        except Exception as e:
            print(f" Ошибка при сохранении: {e}")
    
    def search_documents(self):
        if not self.embeddings_loaded:
            print(" Эмбеддинги не загружены. Сначала обработайте документы.")
            return
        
        print("\n ПОИСК ПО ДОКУМЕНТАМ")
        print(f"Доступно документов: {len(self.searcher.filenames)}")
        print("Введите запрос (или 'назад' для возврата в меню):")
        
        while True:
            try:
                query = input("\n Запрос: ").strip()
                
                if query.lower() in ['назад', 'back', 'exit', 'выход']:
                    break
                
                if not query:
                    print(" Введите текст запроса")
                    continue

                results = self.searcher.search_by_text(
                    query_text=query,
                    embedder=self.embedder,
                    preprocessor=self.preprocessor,
                    top_k=5,
                    threshold=0.1
                )
                self.stats['total_searches'] += 1
                self.stats['queries_history'].append({
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'results_count': len(results)
                })
                if results:
                    print(f"\n РЕзультатов всего  {len(results)}")
                    print("=" * 60)
                    
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}.  {result['filename']}")
                        print(f"    Схожесть: {result['score']:.3%}")
                        print(f"    Текст: {result['text'][:150]}...")
                        print(f"    ID: {result['index']}")

                    avg_score = sum(r['score'] for r in results) / len(results)
                    print(f"\n Средняя схожесть: {avg_score:.3%}")
                    print(f" Лучший результат: {results[0]['score']:.3%}")
                else:
                    print(" Ничего не найдено. ")
                print("Введите следующий запрос или 'назад' для выхода")
                
            except KeyboardInterrupt:
                print("\n Возврат в меню...")
                break
            except Exception as e:
                print(f" Ошибка при поиске: {e}")
    
    def show_statistics(self):
        print("\n СТАТИСТИКА")
        if self.embeddings_loaded:
            print(f" Всего документов: {len(self.searcher.filenames)}")
            print(f" Размерность эмбеддингов: {self.searcher.embeddings.shape[1]}")
        print(f"Всего поисков: {self.stats['total_searches']}")
        if self.stats['queries_history']:
            print("\n История запросов (последние 5):")
            for query in self.stats['queries_history'][-5:]:
                print(f"• {query['query']} ({query['results_count']} результатов)")
        else:
            print(" История запросов: пока пусто")
        
        print("\n Пути к данным:")
        print(f"  Документы: {self.documents_path}")
        print(f"  Эмбеддинги: {self.embeddings_path}")
        print(f"  Метаданные: {self.metadata_path}")
    
    def show_documents_list(self):
        if not self.embeddings_loaded:
            print(" Эмбеддинги не загружены.")
            return
        print("\n СПИСОК ДОКУМЕНТОВ")
        print(f"Всего: {len(self.searcher.filenames)} документов\n")
        
        for i, filename in enumerate(self.searcher.filenames, 1):
            
            preview = self.searcher.texts[i-1][:100].replace('\n', ' ')
            if len(preview) < 100:
                preview = preview.ljust(100)
            else:
                preview = preview[:97] + "..."
            
            print(f"{i:3}. {filename[:30]:30} | {preview}")
    
    def clear_data(self):
        print("\n ОЧИСТКА ДАННЫХ")
        print("Вы уверены, что хотите удалить все эмбеддинги?")
        print("Документы останутся, но нужно будет пересоздать эмбеддинги.")
        
        confirm = input("Введите 'ДА' для подтверждения: ")
        
        if confirm.upper() == 'ДА':
            try:
                if os.path.exists(self.embeddings_path):
                    os.remove(self.embeddings_path)
                    print(f" Удален: {self.embeddings_path}")
                
                if os.path.exists(self.metadata_path):
                    os.remove(self.metadata_path)
                    print(f" Удален: {self.metadata_path}")
                
                self.embeddings_loaded = False
                self.searcher = None
                print(" Все данные удалены. Запустите обработку документов снова.")
                
            except Exception as e:
                print(f" Ошибка при удалении: {e}")
        else:
            print(" Отменено.")
    
    def run(self):
        self.initialize()
        while True:
            try:
                print("  1.  Поиск по документам")
                print("  2.  Обработать документы (создать эмбеддинги)")
                print("  3.  Показать статистику")
                print("  4.  Показать список документов")
                print("  5.  Очистить данные")
                print("  6.  Выход")
                
                choice = input("\n Выберите действие (1-6): ").strip()
                
                if choice == '1':
                    self.search_documents()
                elif choice == '2':
                    self.process_documents()
                elif choice == '3':
                    self.show_statistics()
                elif choice == '4':
                    self.show_documents_list()
                elif choice == '5':
                    self.clear_data()
                elif choice == '6':
                    print("НУ и иди нахуй")
                    break
                else:
                    print("хуйню не делай ")
                    
            except KeyboardInterrupt:
                print("НУ и иди нахуй ")
                break
            except Exception as e:
                print(f" Ошибка: {e}")


def check_dependencies():
   
    required = ['sentence-transformers', 'numpy', 'scikit-learn', 'nltk', 'pymorphy3']
    
    print(" Проверка зависимостей...")
    
    for lib in required:
        try:
            if lib == 'sentence-transformers':
                from sentence_transformers import SentenceTransformer
            elif lib == 'numpy':
                import numpy
            elif lib == 'scikit-learn':
                import sklearn
            elif lib == 'nltk':
                import nltk
            elif lib == 'pymorphy3':
                import pymorphy3
            print(f" {lib}")
        except ImportError:
            print(f" {lib} не установлена")
            return False
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print(" NLTK данные")
    except LookupError:
        print(" NLTK данные не найдены. Скачиваем...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
    
    return True


if __name__ == "__main__":
    if not check_dependencies():
        print("Не все зависимости установлены.")
        print("Установите недостающие библиотеки и запустите снова.")
        sys.exit(1)
    app = SemanticSearchApp()
    app.run()
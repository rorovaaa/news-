import re 
import pymorphy3
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try: 
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SimpleProcessor:
    def clean(self, text):
        text = text.replace('\n', ' ')  
        text = ' '.join(text.split())
        text = text.lower()  
        text = re.sub(r'[^а-яёa-z0-9\s]', ' ', text)
        return text


class AdvancedPreprocessor(SimpleProcessor): 
    def __init__(self , language = 'russian' , use_lemmatization = True):

        super().__init__()
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt_tab')
        
        self.language = language
        self.use_lemmatization = use_lemmatization


        if language == 'russian':
            self.stop_words = set(stopwords.words('russian'))
        else:
            self.stop_words = set(stopwords.words('english'))

        if language == 'russian' and use_lemmatization:
            self.morph_analyzer = pymorphy3.MorphAnalyzer()
        else:
            self.morph_analyzer = None

    def tokenize(self, text):
        clean_text = self.clean(text)
        tokens = word_tokenize(clean_text, language=self.language)
        return tokens
        
    def remove_stopwords(self , tokens):
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens

    def lemmatize(self, tokens):
        if not self.morph_analyzer:
            return tokens
        
        lemmas = []
        for token in tokens:
            parsed = self.morph_analyzer.parse(token)[0]
            lemmas.append(parsed.normal_form)
        return lemmas
    def normalaize(self, text ):
        cleantext = self.clean(text)
        tokens = self.tokenize(cleantext)
        tokens = self.remove_stopwords(tokens)
        if self.use_lemmatization:
            tokens = self.lemmatize(tokens)

        return ' '.join(tokens)
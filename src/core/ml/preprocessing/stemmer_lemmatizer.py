

class StemLemma:
    '''
        desc: Class for stemming and lemmatization of text data
    '''
    def __init__(self, engine: str = 'nltk', language: str = 'english'):
        self.engine = engine
        self.language = language
        if engine == 'nltk':
            from nltk.stem import PorterStemmer, WordNetLemmatizer
        elif engine == 'spacy':
            import spacy
            self.nlp = spacy.load(language)


    def _init_nltk(self):
        '''
            desc:
            input:
            ouptput:
        '''
    
    
    def _init_spacy(self):
        '''
            desc:
            input:
            ouptput:
        '''

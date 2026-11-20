
from nltk.tokenize import word_tokenize
import spacy
from transformers import AutoTokenizer
import sentencepiece as sp

class Tokenizers:
    '''
        desc: Class for all tokenizers
    '''
    def __init__(self, engine: str, model_name: str = None):
        self.engine = engine
        self.model_name = model_name

    
    def tokenize(self, text: str):
        '''
            desc: 
                Tokenizes the input text based on the selected engine
            inpt:
                text [str]: String to be tokenized
            oupt:
                tokens [list[str]]: list of tokens
        '''
        # NLTK
        if self.engine == 'nltk':
            return word_tokenize(text)

        # Spacy
        if self.engine == 'spacy':
            nlp = spacy.load(self.model_name)
            doc = nlp(text)
            tokens = [token.text for token in doc]
            
            # Return
            return tokens

        # Transformers
        if self.engine == 'transformers':
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokens = tokenizer.tokenize(text)
            
            # Return
            return tokens

        # SentencePiece
        if self.engine == 'sentencepiece':
            sp_model = sp.SentencePieceProcessor(model_file = self.model_name)
            tokens = sp_model.encode(text, out_type = str)
            
            # Return
            return tokens


import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from spacy.tokens import Doc

class StemLemma:
    '''
        desc: Class for stemming and lemmatization of text data
    '''
    def __init__(self, engine: str = 'nltk', language: str = 'en_core_web_sm'):
        self.engine = engine
        self.language = language
        if engine == 'nltk':
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        elif engine == 'spacy':
            self.nlp = spacy.load(language)


    def stem(self, text: list[str], join: bool = True) -> str | list[str]:
        '''
            desc:
                Given a list of tokenized text, stem each token
            input:
                text [list[str]]: list of tokenized text
                join [bool]: whether to join the list into a single string
            output:
                stems [str | list[str]]: stemmed text as a string or list
        '''
        # Get stems
        stems = [self.stemmer.stem(token) for token in text]

        # Return
        if join:
            return ' '.join(stems)
        else:
            return stems


    def get_wordnet_pos(self, word: str) -> str:
        '''
            desc:
                Helper function to map pos tag to the wordnet format
                from nltk for lemmatization
            inpt:
                word [str]: word to get pos tag for
            oupt:
                [str]: wordnet pos tag
        '''
        # Get tag
        tag = nltk.pos_tag([word])[0][1][0].upper()

        # Mapping for part of speech
        pos_map = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        # Return
        return pos_map.get(tag, wordnet.NOUN)

    
    def lemmatize(self, text: list[str] | str, join: bool = True) -> str | list[str]:
        '''
            desc:
                Given a list of tokenized text, lemmatize each token
            input:
                text [list[str] | str]: list of tokenized text or a single string
                join [bool]: whether to join the list into a single string
            output:
                lemmas [str | list[str]]: lemmatized text as a string or list
        '''
        if self.engine == 'nltk':
            # Get all lemmatized tokens
            tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in text]
            
            # Return
            if join:
                return ' '.join(tokens)
            else:
                return tokens

        if self.engine == 'spacy':
            # Create spacy doc
            doc = self.nlp(text)
            
            # Return
            if join:
                return ' '.join([token.lemma_ for token in doc])
            else:
                return [token.lemma_ for token in doc]
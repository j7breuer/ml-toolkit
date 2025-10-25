

class StemLemma:
    '''
        desc: Class for stemming and lemmatization of text data
    '''
    def __init__(self, engine: str = 'nltk', language: str = 'en_core_web_sm'):
        self.engine = engine
        self.language = language
        if engine == 'nltk':
            import nltk
            from nltk.corpus import wordnet
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        elif engine == 'spacy':
            import spacy
            from spacy.tokens import Doc
            self.nlp = spacy.load(language)


    def stem(self, text: list[str], join: bool = True) -> str | list[str]:
        '''
            desc:
            input:
            output:
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
            inpt:
            oupt:
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
            input:
            output:
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
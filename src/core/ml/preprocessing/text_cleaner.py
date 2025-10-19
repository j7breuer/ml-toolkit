
import re
import emoji
import string
import nltk
import regex
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Define url pattern
URL_PATTERN = re.compile(
    r'\b(?:https?://|www\.)[\w.-]+\.[a-zA-Z]{2,}(?:[/?#]\S*)?'
    r'|\b[\w.-]+\.[a-zA-Z]{2,}(?:[/?#]\S*)?'
)

class TextCleaner:
    def __init__(self, language = 'english'):
        nltk.download('stopwords')
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
    @staticmethod
    def convert_case(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Return
        return text.lower()
    
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''        
        # Revmove unicode
        cleaned_text = regex.sub(r'\p{P}+', '', cleaned_text)

        # Return
        return cleaned_text
    
    
    @staticmethod
    def remove_special_characters(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Return
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    

    @staticmethod
    def remove_emojis(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Return
        return emoji.replace_emoji(text, replace = '')


    @staticmethod
    def normalize_whitespace(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Return
        return re.sub(r'\s+', ' ', text).strip()

    
    @staticmethod
    def replace_regex_pattern(text: str, pattern: str, replacement_value: str = '') -> str:
        '''
            desc:
            input:
            output:
        '''
        # Return
        return re.sub(pattern, replacement_value, text)
    

    @staticmethod
    def remove_urls(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''        
        # Return
        return re.sub(URL_PATTERN, '', text, flags=re.IGNORECASE)
    

    @staticmethod
    def remove_html_css(text: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Apply html parser
        soup = BeautifulSoup(text, "html.parser")

        # Loop through style/script tags and remove them
        for script_style in soup(["style", "script"]):
            script_style.decompose()

        # Get text
        cleaned_text = soup.get_text(separator = " ")
        cleaned_text = " ".join(cleaned_text.split())

        # Return
        return cleaned_text
    

    def remove_stopwords(self, text: str) -> str:
        '''
            desc:
            input:
            output:
        '''        
        # Tokenize text
        tokens = text.lower().split()

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Return
        return ' '.join(filtered_tokens)
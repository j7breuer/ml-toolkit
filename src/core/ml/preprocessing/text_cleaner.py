
import re
import emoji
import string
import nltk
import regex
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from src.config.settings import RegexPatterns

class TextCleaner:
    '''
        desc: Class for cleaning text values
    '''
    def __init__(self, language = 'english'):
        nltk.download('stopwords')
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
    @staticmethod
    def clean_financial_strings(text: str) -> str:
        '''
            desc:
                Given a string financial value, clean text values 
            input:
                text [str]: Value to clean
            output:
                [str]: financial strings removed
        '''        
        # Return
        return re.sub(RegexPatterns.FINANCIAL_STRINGS, '', text)
    
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        '''
            desc:
                Function specific to removing punctuation only
            input:
                text [str]: Value to clean
            output:
                cleaned_text [str]: Cleaned value
        '''        
        # Revmove unicode
        cleaned_text = regex.sub(r'\p{P}+', '', cleaned_text)

        # Return
        return cleaned_text
    
    
    @staticmethod
    def remove_special_characters(text: str) -> str:
        '''
            desc:
                Function specific to removing special characters
            input:
                text [str]: Value to clean
            output:
                [str]: Cleaned values
        '''
        # Return
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    

    @staticmethod
    def remove_emojis(text: str) -> str:
        '''
            desc:
                Function specific to removing emojis characters
            input:
                text [str]: Value to clean
            output:
                [str]: Cleaned values
        '''
        # Return
        return emoji.replace_emoji(text, replace = '')


    @staticmethod
    def normalize_whitespace(text: str) -> str:
        '''
            desc:
                Functino for normalizing all whitespace into one space
            input:
                text [str]: Value to clean
            output:
                [str]: Cleaned values
        '''
        # Return
        return re.sub(r'\s+', ' ', text).strip()

    
    @staticmethod
    def replace_regex_pattern(text: str, pattern: str, replacement_value: str = '') -> str:
        '''
            desc:
                Given a custom regex pattern, replace it with input value
            input:
                text [str]: string to clean
                pattern [str]: regex pattern to apply
                replacement value [str]: default to empty string, value to replace with
            output:
                [str]: String with replaced value
        '''
        # Return
        return re.sub(pattern, replacement_value, text)
    

    @staticmethod
    def remove_urls(text: str) -> str:
        '''
            desc:
                Given a text value, use regex url pattern to remove all
            input:
                text [str]: text value to search and remove urls
            output:
                [str]: Cleaned str
        '''        
        # Return
        return re.sub(RegexPatterns.URL, '', text)
    

    @staticmethod
    def remove_html_css(text: str) -> str:
        '''
            desc:
                Remove html and css from text
            input:
                text [str]: Text value to remove html/css from
            output:
                cleaned_text [str]: Cleaned string
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
                Given a string, remove stopwords based on language/model
            input:
                text [str]: String to remove stop words
            output:
                [str]: Cleaned string joined back
        '''        
        # Tokenize text
        tokens = text.lower().split()

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Return
        return ' '.join(filtered_tokens)
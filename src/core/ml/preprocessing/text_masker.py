
import re
from src.config.settings import RegexPatterns

class TextMasker:
    '''
        desc: A class for masking common patterns in text data
    '''
    @staticmethod
    def mask_email(text: str, replacement_value: str = '<EMAIL>') -> str:
        '''
            desc:
            input:
            output:
        '''
        # Mask email addresses
        return re.sub(RegexPatterns.EMAIL, replacement_value, text)


    @staticmethod
    def mask_phone_number(text: str, replacement_value: str = '<PHONE>') -> str:
        '''
            desc:
            input:
            output:
        '''
        # Mask phone numbers
        return re.sub(RegexPatterns.PHONE, replacement_value, text)

    

    @staticmethod
    def mask_url(text: str, replacement_value: str = '<URL>') -> str:
        '''
            desc:
            input:
            output:
        '''
        # Mask URLs
        return re.sub(RegexPatterns.URL, replacement_value, text)
    

    @staticmethod
    def mask_custom_value(text: str, pattern: str, replacement_mask: str) -> str:
        '''
            desc:
            input:
            output:
        '''
        # Mask custom patterns
        return re.sub(pattern, replacement_mask, text)

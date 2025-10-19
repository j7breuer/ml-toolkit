
import re

class RegexPatterns:
    EMAIL = re.compile(
        r'\b[\w\.-]+@[\w\.-]+\.\w+\b', 
        flags = re.IGNORECASE
    )
    PHONE = re.compile(
        r'(\+?\d{1,2}[\s\-\.]?)?(\(?\d{3}\)?[\s\-\.]?)?\d{3}[\s\-\.]?\d{4}',
        flags = re.IGNORECASE
    )
    URL = re.compile(
        r'\b(?:https?://|www\.)[\w.-]+\.[a-zA-Z]{2,}(?:[/?#]\S*)?'
        r'|\b[\w.-]+\.[a-zA-Z]{2,}(?:[/?#]\S*)?',
        flags = re.IGNORECASE
    )
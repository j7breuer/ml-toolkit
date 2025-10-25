

import nltk
from src.config.settings import NltkSettings


class NltkHelper:
    '''
        desc: Helper class for NLTK resource management
    '''
    @staticmethod
    def _ensure_nltk_resource(resource: str, resource_paths = None, quiet = True):
        '''
            desc:
            input:
            output:
        '''
        # Search through paths in settings or provided
        for path in (resource_paths or NltkSettings.DEFAULT_INSTALL_PATHS):
            try:
                nltk.data.find(f'{path}/{resource}')
                return
            except LookupError:
                continue

        # Download resource if not found
        nltk.download(resource, quiet = quiet)
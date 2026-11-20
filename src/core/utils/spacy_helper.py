

import spacy

# Set cached models
_cached_models = {}

class SpacyHelper:
    '''
        desc: Helper class for loading and managing spaCy models
    '''
    @staticmethod
    def _ensure_spacy_model(model_name: str = 'en_core_web_sm'):
        '''
            desc:
            input:
            output:
        '''
        # Check in cache
        if model_name in _cached_models:
            # Return if found
            return _cached_models[model_name]
        
        try:
            # Check to see if can load
            nlp = spacy.load(model_name)
        except OSError:
            # Download and load model if not found
            from spacy.cli import download
            download(model_name)
            nlp = spacy.load(model_name)
        
        # Add to cached models
        _cached_models[model_name] = nlp
        
        # Return model
        return nlp
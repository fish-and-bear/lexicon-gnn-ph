class InvalidLanguageCode(Exception):
    """Exception raised when an invalid language code is provided."""
    pass

class InvalidLanguageMapping(Exception):
    """Exception raised when an invalid language mapping is provided."""
    pass

class LanguageType:
    """Base class for language types."""
    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    def __eq__(self, other):
        if not isinstance(other, LanguageType):
            return False
        return self.code == other.code
    
    def __hash__(self):
        return hash(self.code)

class ScriptType:
    """Base class for script types."""
    def __init__(self, code: str, name: str):
        self.code = code
        self.name = name
    
    def __str__(self):
        return f"{self.name} ({self.code})"
    
    def __eq__(self, other):
        if not isinstance(other, ScriptType):
            return False
        return self.code == other.code
    
    def __hash__(self):
        return hash(self.code) 
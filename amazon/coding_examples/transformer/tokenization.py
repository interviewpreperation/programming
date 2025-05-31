import re
from typing import List, Union

class Tokenizer:
    def __init__(self, 
                 lower_case: bool = True, 
                 remove_punctuation: bool = False,
                 split_on_whitespace: bool = True,
                 custom_pattern: str = None):
        """
        Initialize the tokenizer with configuration options.
        
        Args:
            lower_case: Convert all tokens to lowercase
            remove_punctuation: Remove punctuation from tokens
            split_on_whitespace: Split on whitespace (if False, uses custom pattern)
            custom_pattern: Regex pattern for tokenization if not splitting on whitespace
        """
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.split_on_whitespace = split_on_whitespace
        self.custom_pattern = custom_pattern or r'\w+'
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text according to the configured rules.
        
        Args:
            text: Input string to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Apply lowercase if configured
        if self.lower_case:
            text = text.lower()
            
        # Tokenize based on configuration
        if self.split_on_whitespace:
            tokens = text.split()
        else:
            tokens = re.findall(self.custom_pattern, text)
            
        # Remove punctuation if configured
        if self.remove_punctuation:
            tokens = [self._remove_punctuation(token) for token in tokens]
            
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
        
    def _remove_punctuation(self, token: str) -> str:
        """Helper method to remove punctuation from a token."""
        return re.sub(r'[^\w\s]', '', token)
    
    def get_vocabulary(self, texts: Union[str, List[str]]) -> dict:
        """
        Build a vocabulary from text or list of texts.
        
        Args:
            texts: Either a single string or list of strings
            
        Returns:
            Dictionary with tokens as keys and their frequencies as values
        """
        if isinstance(texts, str):
            texts = [texts]
            
        vocab = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                vocab[token] = vocab.get(token, 0) + 1
                
        return vocab


# Example usage
if __name__ == "__main__":
    sample_text = "Hello, world! This is a test. Tokenization is important for NLP."
    
    # Basic tokenizer (splits on whitespace)
    print("Basic tokenizer:")
    basic_tokenizer = Tokenizer()
    print(basic_tokenizer.tokenize(sample_text))
    
    # Tokenizer that removes punctuation
    print("\nTokenizer with punctuation removal:")
    no_punct_tokenizer = Tokenizer(remove_punctuation=True)
    print(no_punct_tokenizer.tokenize(sample_text))
    
    # Tokenizer using regex pattern (words only)
    print("\nRegex word tokenizer:")
    regex_tokenizer = Tokenizer(split_on_whitespace=False)
    print(regex_tokenizer.tokenize(sample_text))

    # Building a vocabulary
    print("\nVocabulary:")
    vocab = no_punct_tokenizer.get_vocabulary(sample_text)
    print(vocab)


'''
Key Features Demonstrated:
Configurable Tokenization: The class can be configured for different tokenization needs (case sensitivity, punctuation handling, etc.)

Multiple Tokenization Methods: Supports both whitespace splitting and regex-based tokenization

Vocabulary Building: Includes a method to build a frequency vocabulary from texts

Clean Code Structure: Well-organized with helper methods and clear documentation

Type Hints: Uses Python type hints for better code clarity

Edge Case Handling: Properly handles empty input and other edge cases

Example Usage: Demonstrates how to use the class with different configurations

During an interview, you could walk through:

The design decisions

Trade-offs between different tokenization approaches

How you might extend this (e.g., adding stemming/lemmatization)

Performance considerations

Unicode handling (which this basic version doesn't fully address)
'''
    


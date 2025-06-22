# Shredword User Documentation

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)
8. [Examples](#examples)

## Installation

Install Shredword using pip:

```bash
pip install shred
```

### Requirements

- Python 3.7+
- C compiler (for optimal performance)
- Internet connection (for downloading vocabulary files)

## Quick Start

```python
from shred import load_encoding

# Load a tokenizer with a specific encoding
tokenizer = load_encoding("gpt2")

# Basic tokenization
text = "Hello, world! This is a test."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")
```

## Core Concepts

### Tokenization

Tokenization is the process of converting text into numerical tokens that can be processed by machine learning models. shred uses Byte Pair Encoding (BPE) for efficient subword tokenization.

### Vocabularies

Each encoding comes with a pre-trained vocabulary that maps text pieces to token IDs. shred automatically downloads these vocabularies from the official repository.

### Special Tokens

Special tokens are reserved tokens with specific meanings (e.g., `<|endoftext|>`, `<pad>`, `<unk>`). They can be handled specially during encoding and decoding.

## API Reference

### Loading Tokenizers

#### `load_encoding(encoding_name: str) -> Shred`

Creates and loads a tokenizer with the specified encoding.

**Parameters:**

- `encoding_name` (str): Name of the encoding to load (e.g., "gpt2", "gpt4")

**Returns:**

- `Shred`: Initialized tokenizer instance

**Example:**

```python
tokenizer = load_encoding("pre_16k")
```

### Shred Class

The main tokenizer class providing encoding and decoding functionality.

#### Methods

##### `encode(text: str, allowed_special: Optional[List[str]] = None) -> List[int]`

Encodes text into a list of token IDs.

**Parameters:**

- `text` (str): Input text to tokenize
- `allowed_special` (Optional[List[str]]): List of special tokens to allow, or "all" for all special tokens

**Returns:**

- `List[int]`: List of token IDs

**Example:**

```python
tokens = tokenizer.encode("Hello world!")
tokens_with_special = tokenizer.encode("Hello <|endoftext|>", allowed_special=["<|endoftext|>"])
```

##### `encode_ordinary(text: str) -> List[int]`

Encodes text without processing any special tokens.

**Parameters:**

- `text` (str): Input text to tokenize

**Returns:**

- `List[int]`: List of token IDs

**Example:**

```python
tokens = tokenizer.encode_ordinary("Hello world!")
```

##### `decode(tokens: List[int]) -> str`

Decodes a list of token IDs back to text.

**Parameters:**

- `tokens` (List[int]): List of token IDs to decode

**Returns:**

- `str`: Decoded text

**Example:**

```python
text = tokenizer.decode([15496, 11, 995, 0])
```

##### `encode_with_unstable(text: str, allowed_special: Optional[List[str]] = None) -> Dict`

Advanced encoding method that returns both tokens and potential completions.

**Parameters:**

- `text` (str): Input text to tokenize
- `allowed_special` (Optional[List[str]]): List of special tokens to allow

**Returns:**

- `Dict`: Dictionary with 'tokens' and 'completions' keys

**Example:**

```python
result = tokenizer.encode_with_unstable("Hello world")
tokens = result['tokens']
completions = result['completions']
```

#### Properties

##### `vocab_size: int`

Returns the size of the vocabulary.

```python
print(f"Vocabulary size: {tokenizer.vocab_size}")
```

##### `special_tokens: Dict[str, int]`

Returns a dictionary of special tokens and their IDs.

```python
special = tokenizer.special_tokens
print(f"Special tokens: {special}")
```

##### `vocab: List[str]`

Returns the complete vocabulary as a list of strings.

```python
vocabulary = tokenizer.vocab
```

##### `encoder: Dict[bytes, int]`

Returns the encoder mapping from bytes to token IDs.

```python
encoder_map = tokenizer.encoder
```

##### `decoder: Dict[int, bytes]`

Returns the decoder mapping from token IDs to bytes.

```python
decoder_map = tokenizer.decoder
```

## Advanced Usage

### Working with Special Tokens

```python
# Allow specific special tokens
tokens = tokenizer.encode("Text with <|endoftext|>", allowed_special=["<|endoftext|>"])

# Allow all special tokens
tokens = tokenizer.encode("Text with specials", allowed_special="all")

# Get all available special tokens
special_tokens = tokenizer.special_tokens
print(f"Available special tokens: {list(special_tokens.keys())}")
```

### Batch Processing

```python
texts = ["First text", "Second text", "Third text"]
all_tokens = []

for text in texts:
    tokens = tokenizer.encode(text)
    all_tokens.append(tokens)

# Decode all at once
decoded_texts = [tokenizer.decode(tokens) for tokens in all_tokens]
```

### Custom Vocabulary Inspection

```python
# Inspect vocabulary
vocab = tokenizer.vocab
print(f"First 10 tokens: {vocab[:10]}")

# Find token ID for specific text
text_piece = "hello"
text_bytes = text_piece.encode('utf-8')
if text_bytes in tokenizer.encoder:
    token_id = tokenizer.encoder[text_bytes]
    print(f"Token ID for '{text_piece}': {token_id}")
```

## Error Handling

shred includes robust error handling and fallback mechanisms:

```python
try:
    tokenizer = load_encoding("nonexistent_encoding")
except ValueError as e:
    print(f"Failed to load encoding: {e}")

try:
    tokens = tokenizer.encode("Some text")
except RuntimeError as e:
    print(f"Encoding failed: {e}")
```

### Common Errors

- **ValueError**: Invalid encoding name or corrupted vocabulary file
- **RuntimeError**: Tokenizer not initialized or C library issues
- **UnicodeError**: Text encoding/decoding issues

## Performance Tips

1. **Reuse Tokenizers**: Create tokenizer instances once and reuse them
2. **Batch Processing**: Process multiple texts in batches when possible
3. **Avoid Special Tokens**: Use `encode_ordinary()` when special tokens aren't needed
4. **Memory Management**: The library handles memory management automatically

```python
# Good: Reuse tokenizer
tokenizer = load_encoding("gpt2")
for text in texts:
    tokens = tokenizer.encode(text)

# Less efficient: Create new tokenizer each time
for text in texts:
    tokenizer = load_encoding("gpt2")  # Avoid this
    tokens = tokenizer.encode(text)
```

## Examples

### Basic Text Processing

```python
from shred import load_encoding

# Load tokenizer
tokenizer = load_encoding("gpt2")

# Process a document
document = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language.
"""

# Tokenize
tokens = tokenizer.encode(document)
print(f"Number of tokens: {len(tokens)}")
print(f"First 10 tokens: {tokens[:10]}")

# Check vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")
```

### Working with Code

```python
# Tokenizing code
code = """
def hello_world():
    print("Hello, world!")
    return True
"""

tokens = tokenizer.encode(code)
decoded = tokenizer.decode(tokens)
print(f"Original matches decoded: {code == decoded}")
```

### Analyzing Token Distribution

```python
import collections

# Analyze token frequency in a text
text = "The quick brown fox jumps over the lazy dog. " * 100
tokens = tokenizer.encode(text)

# Count token frequencies
token_counts = collections.Counter(tokens)
most_common = token_counts.most_common(10)

print("Most common tokens:")
for token_id, count in most_common:
    token_text = tokenizer.decode([token_id])
    print(f"Token {token_id} ('{token_text}'): {count} times")
```

### Handling Different Languages

```python
# Multilingual text
texts = [
    "Hello, world!",           # English
    "¡Hola, mundo!",          # Spanish
    "Bonjour, le monde!",     # French
    "こんにちは、世界！",          # Japanese
]

for text in texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {text == decoded}")
    print("-" * 50)
```

### Comparing Encodings

```python
# Compare different encodings (if available)
encodings = ["pre_16k", "pre_25k"]
text = "This is a sample text for comparison."

for encoding_name in encodings:
    try:
        tokenizer = load_encoding(encoding_name)
        tokens = tokenizer.encode(text)
        print(f"{encoding_name}: {len(tokens)} tokens - {tokens}")
    except ValueError:
        print(f"{encoding_name}: Not available")
```

This documentation provides comprehensive coverage of the shred library's functionality. For additional help or questions, please refer to the project's GitHub repository.

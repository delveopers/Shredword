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

### BPE Tokenization

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

### Word Tokenization

```python
from shred import WordTokenizer

# Create a word tokenizer
tokenizer = WordTokenizer()

# Tokenize text into words
text = "Hello, world! This is a test."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
```

### TF-IDF Embeddings

```python
from shred import TfidfEmbedding

# Create embedding model
embedding = TfidfEmbedding()

# Add training documents
documents = [
  "The quick brown fox jumps over the lazy dog",
  "Python programming is fun and exciting"
]
embedding.add_documents(documents)

# Encode text
ids = embedding.encode_ids("The lazy fox")
dense_vec = embedding.encode_tfidf_dense("The lazy fox")
indices, values = embedding.encode_tfidf_sparse("The lazy fox")

# Save and load vocabulary
embedding.save("vocab.txt")
loaded = TfidfEmbedding.load("vocab.txt")
```

## Core Concepts

### Tokenization

Tokenization is the process of converting text into numerical tokens that can be processed by machine learning models. Shredword supports multiple tokenization methods:

- **BPE (Byte Pair Encoding)**: Efficient subword tokenization for neural models
- **Word Tokenization**: Fast word-level tokenization with contraction handling

### Vocabularies

Each encoding comes with a pre-trained vocabulary that maps text pieces to token IDs. Shredword automatically downloads these vocabularies from the official repository.

### Special Tokens

Special tokens are reserved tokens with specific meanings (e.g., `<|endoftext|>`, `<pad>`, `<unk>`). They can be handled specially during encoding and decoding.

### TF-IDF Embeddings

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. Shredword provides both dense and sparse TF-IDF representations.

## API Reference

### Loading Tokenizers

#### `load_encoding(encoding_name: str) -> Shred`

Creates and loads a BPE tokenizer with the specified encoding.

**Parameters:**

- `encoding_name` (str): Name of the encoding to load (e.g., "gpt2", "gpt4")

**Returns:**

- `Shred`: Initialized tokenizer instance

**Example:**

```python
tokenizer = load_encoding("pre_16k")
```

### Shred Class

The main BPE tokenizer class providing encoding and decoding functionality.

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

### WordTokenizer Class

Fast word-level tokenizer with support for contractions and punctuation.

#### Methods

##### `tokenize(text: str) -> List[str]`

Tokenizes text into words.

**Parameters:**

- `text` (str): Input text to tokenize

**Returns:**

- `List[str]`: List of word tokens

**Example:**

```python
tokenizer = WordTokenizer()
tokens = tokenizer.tokenize("Hello, world! This is a test.")
# Output: ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
```

**Features:**

- Handles contractions (e.g., "don't" → "do", "n't")
- Preserves punctuation as separate tokens
- Handles decimal numbers (e.g., "3.14")
- Treats hyphens in compound words (e.g., "self-driving")

### TfidfEmbedding Class

TF-IDF vectorization with vocabulary management.

#### Methods

##### `add_document(text: str) -> None`

Adds a document to build vocabulary and IDF statistics.

**Parameters:**

- `text` (str): Document text to add

**Example:**

```python
embedding = TfidfEmbedding()
embedding.add_document("The quick brown fox")
```

##### `add_documents(documents: List[str]) -> None`

Adds multiple documents at once.

**Parameters:**

- `documents` (List[str]): List of document texts

**Example:**

```python
embedding.add_documents([
  "First document text",
  "Second document text"
])
```

##### `encode_ids(text: str) -> List[int]`

Encodes text as token IDs.

**Parameters:**

- `text` (str): Input text to encode

**Returns:**

- `List[int]`: List of token IDs

**Example:**

```python
ids = embedding.encode_ids("The quick fox")
```

##### `encode_tfidf_dense(text: str) -> np.ndarray`

Returns dense TF-IDF vector representation.

**Parameters:**

- `text` (str): Input text to encode

**Returns:**

- `np.ndarray`: Dense vector of shape (vocab_size,)

**Example:**

```python
vector = embedding.encode_tfidf_dense("The quick fox")
print(f"Vector shape: {vector.shape}")
```

##### `encode_tfidf_sparse(text: str) -> Tuple[List[int], List[float]]`

Returns sparse TF-IDF representation.

**Parameters:**

- `text` (str): Input text to encode

**Returns:**

- `Tuple[List[int], List[float]]`: (indices, values) for non-zero elements

**Example:**

```python
indices, values = embedding.encode_tfidf_sparse("The quick fox")
print(f"Non-zero indices: {indices}")
print(f"TF-IDF values: {values}")
```

##### `save(path: str) -> Optional[str]`

Saves vocabulary to file.

**Parameters:**

- `path` (str): File path to save to

**Returns:**

- `Optional[str]`: Error message if failed, None if successful

**Example:**

```python
error = embedding.save("vocab.txt")
if error:
  print(f"Save failed: {error}")
```

##### `load(path: str) -> TfidfEmbedding` (static method)

Loads vocabulary from file.

**Parameters:**

- `path` (str): File path to load from

**Returns:**

- `TfidfEmbedding`: Loaded embedding instance

**Example:**

```python
embedding = TfidfEmbedding.load("vocab.txt")
```

##### `lookup(token: str) -> int`

Looks up token ID in vocabulary.

**Parameters:**

- `token` (str): Token to look up

**Returns:**

- `int`: Token ID, or -1 if not found

**Example:**

```python
token_id = embedding.lookup("quick")
```

##### `get_vocabulary() -> Dict[str, Tuple[int, int]]`

Returns complete vocabulary with statistics.

**Returns:**

- `Dict[str, Tuple[int, int]]`: {token: (count, document_frequency)}

**Example:**

```python
vocab = embedding.get_vocabulary()
for token, (count, df) in vocab.items():
  print(f"{token}: count={count}, df={df}")
```

#### Properties

##### `vocab_size: int`

Returns the size of the vocabulary.

```python
print(f"Vocabulary size: {embedding.vocab_size}")
```

##### `document_count: int`

Returns the number of documents processed.

```python
print(f"Documents: {embedding.document_count}")
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

### Building TF-IDF Models

```python
from shred import TfidfEmbedding

# Create and train model
embedding = TfidfEmbedding()

# Add training corpus
corpus = [
  "Natural language processing is a field of AI",
  "Machine learning models process data efficiently",
  "Deep learning uses neural networks for complex tasks"
]
embedding.add_documents(corpus)

# Analyze new text
test_text = "AI and machine learning are powerful"
indices, values = embedding.encode_tfidf_sparse(test_text)

# Map indices back to tokens
vocab = embedding.get_vocabulary()
vocab_list = list(vocab.keys())
for idx, val in zip(indices, values):
  if idx < len(vocab_list):
    print(f"{vocab_list[idx]}: {val:.4f}")
```

### Comparing Word vs BPE Tokenization

```python
from shred import load_encoding, WordTokenizer

text = "The quick brown fox jumps over the lazy dog"

# Word tokenization
word_tokenizer = WordTokenizer()
word_tokens = word_tokenizer.tokenize(text)
print(f"Word tokens ({len(word_tokens)}): {word_tokens}")

# BPE tokenization
bpe_tokenizer = load_encoding("pre_16k")
bpe_tokens = bpe_tokenizer.encode(text)
print(f"BPE tokens ({len(bpe_tokens)}): {bpe_tokens}")
```

## Error Handling

Shredword includes robust error handling and fallback mechanisms:

```python
# BPE tokenizer errors
try:
    tokenizer = load_encoding("nonexistent_encoding")
except ValueError as e:
    print(f"Failed to load encoding: {e}")

try:
    tokens = tokenizer.encode("Some text")
except RuntimeError as e:
    print(f"Encoding failed: {e}")

# TF-IDF embedding errors
try:
    embedding = TfidfEmbedding.load("missing_file.txt")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"Load failed: {e}")
```

### Common Errors

- **ValueError**: Invalid encoding name or corrupted vocabulary file
- **RuntimeError**: Tokenizer not initialized or C library issues
- **FileNotFoundError**: Vocabulary file not found during load
- **UnicodeError**: Text encoding/decoding issues

## Performance Tips

1. **Reuse Tokenizers**: Create tokenizer instances once and reuse them
2. **Batch Processing**: Process multiple texts in batches when possible
3. **Avoid Special Tokens**: Use `encode_ordinary()` when special tokens aren't needed
4. **Sparse Representations**: Use sparse TF-IDF for large vocabularies
5. **Memory Management**: The library handles memory management automatically

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

### Document Similarity with TF-IDF

```python
from shred import TfidfEmbedding
import numpy as np

# Create embedding model
embedding = TfidfEmbedding()

# Training documents
documents = [
  "Machine learning is a subset of artificial intelligence",
  "Deep learning uses neural networks with many layers",
  "Natural language processing deals with human language",
  "Computer vision enables machines to understand images"
]
embedding.add_documents(documents)

# Compute similarity between two texts
def cosine_similarity(vec1, vec2):
  dot = np.dot(vec1, vec2)
  norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
  return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

text1 = "Neural networks for deep learning"
text2 = "Language processing with AI"

vec1 = embedding.encode_tfidf_dense(text1)
vec2 = embedding.encode_tfidf_dense(text2)

similarity = cosine_similarity(vec1, vec2)
print(f"Similarity between texts: {similarity:.4f}")
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

### Word Tokenization with Contractions

```python
from shred import WordTokenizer

tokenizer = WordTokenizer()

# Handles contractions properly
text = "I don't think we'll be there. It's a long way."
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['I', 'do', "n't", 'think', 'we', "'ll", 'be', 'there', '.', ...]
```

This documentation provides comprehensive coverage of the Shredword library's functionality. For additional help or questions, please refer to the project's GitHub repository.

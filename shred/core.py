"""
  @file core.py
  @brief High-level Python interface for ShredBPE tokenizer with SentencePiece integration

  * This module provides a clean, Pythonic interface to the ShredBPE tokenizer,
  handling memory management, error checking, and type conversions automatically.
  * Now includes direct loading from SentencePiece-trained vocabularies.
"""

from typing import List, Dict, Optional, Tuple, Union
import json, base64, requests
from pathlib import Path
from ctypes import byref, POINTER, c_size_t
from cbase import *

class ShredBPEError(Exception):
  """Exception raised by ShredBPE operations"""

  ERROR_MESSAGES = {
    ShredError.ERROR_NULL_POINTER: "Null pointer error",
    ShredError.ERROR_MEMORY_ALLOCATION: "Memory allocation failed",
    ShredError.ERROR_INVALID_TOKEN: "Invalid token",
    ShredError.ERROR_REGEX_COMPILE: "Regex compilation failed",
    ShredError.ERROR_REGEX_MATCH: "Regex match failed",
    ShredError.ERROR_INVALID_UTF8: "Invalid UTF-8 encoding"
  }
  
  def __init__(self, error_code: int, message: str = None):
    self.error_code = error_code
    if message is None:
      message = self.ERROR_MESSAGES.get(error_code, f"Unknown error code: {error_code}")
    super().__init__(message)

class CompletionResult:
  """Represents completion possibilities for partial tokenization"""

  def __init__(self, tokens: List[int], completions: List[List[int]]):
    self.tokens = tokens
    self.completions = completions

  def __repr__(self):
    return f"CompletionResult(tokens={self.tokens}, completions={len(self.completions)} possibilities)"

DEFAULT_REGEX = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
DEFAULT_REPO_URL = "https://raw.githubusercontent.com/delveopers/shredword/main/vocabs"
  
class Shred:
  """High-level Python interface to ShredBPE tokenizer with SentencePiece support"""

  def __init__(self):
    """Initialize the tokenizer interface"""
    self._interface = ShredBPEInterface(None)
    self._bpe_ptr = None
    self._is_initialized = False
    self.repo_url = DEFAULT_REPO_URL

  def load_from_encoding(self, encoding_name: str, pattern: str = DEFAULT_REGEX) -> None:
    """Load vocabulary from vocab file by encoding name (no SentencePiece dependency)

    Args:
      encoding_name: Name of the encoding (e.g., 'base_50k', 'ava_v1', 'pre_16k')
      pattern: Regex pattern for text preprocessing
    """
    vocab_url = f"{self.repo_url}/{encoding_name}.vocab"
    try:
      # Download the vocab file
      vocab_response = requests.get(vocab_url, timeout=30)
      vocab_response.raise_for_status()
      vocab_text = vocab_response.text
      encoder_data, special_tokens = self._parse_vocab_file(vocab_text) # Parse the vocab file
      self.load_vocab(encoder_data, special_tokens, pattern)  # Load vocabulary into the tokenizer

    except requests.RequestException as e:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION, f"Failed to download vocab from {vocab_url}: {e}")
    except Exception as e:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION, f"Failed to parse vocab file: {e}")

  def _parse_vocab_file(self, vocab_text: str) -> Tuple[Dict[bytes, int], Dict[str, int]]:
    """Parse SentencePiece vocab file format
    
    Args:
      vocab_text: Content of the .vocab file
      
    Returns:
      Tuple of (encoder_data, special_tokens)
    """
    encoder_data, special_tokens = {}, {}
    for line_num, line in enumerate(vocab_text.strip().split('\n')):
      line = line.strip()
      if not line: continue
      # Split on tab - vocab files have format: "piece\tscore"
      parts = line.split('\t')
      if len(parts) < 2: continue
      piece, token_id = parts[0], line_num
      try:
        # Handle special tokens (those starting with < and ending with >)
        if piece.startswith('<') and piece.endswith('>'):
          special_tokens[piece] = token_id
        elif piece.startswith('▁'):
          # SentencePiece uses ▁ to represent spaces
          actual_piece = piece.replace('▁', ' ')
          encoder_data[actual_piece.encode('utf-8')] = token_id
        else:
          # Handle byte-level tokens
          if piece.startswith('<0x') and piece.endswith('>'):
            # Byte token format like <0x20>
            hex_val = piece[3:-1]
            byte_val = bytes([int(hex_val, 16)])
            encoder_data[byte_val] = token_id
          else:
            # Regular token - encode as UTF-8 bytes
            encoder_data[piece.encode('utf-8')] = token_id
      except (ValueError, UnicodeEncodeError):
        # Fallback for problematic tokens
        try:
          encoder_data[piece.encode('utf-8', errors='replace')] = token_id
        except:
          # Skip tokens that can't be processed
          continue    
    return encoder_data, special_tokens

  def load_vocab(self, encoder_data: Dict[bytes, int], special_tokens: Optional[Dict[str, int]] = None, pattern: str = DEFAULT_REGEX) -> None:
    """Load vocabulary and initialize the tokenizer

    Args:
      encoder_data: Dictionary mapping byte sequences to token ranks
      special_tokens: Dictionary mapping special token strings to ranks
      pattern: Regex pattern for text preprocessing
    """
    if self._is_initialized: self.cleanup()

    # Prepare encoder data
    encoder_keys, encoder_key_lens, encoder_values = [], [], []
    for byte_seq, rank in encoder_data.items():
      if isinstance(byte_seq, str):
        byte_seq = byte_seq.encode('utf-8')
      encoder_keys.append(byte_seq)
      encoder_key_lens.append(len(byte_seq))
      encoder_values.append(rank)
    # Convert to C arrays
    c_encoder_keys = []
    for key in encoder_keys:
      c_key, _ = create_byte_array_from_bytes(key)
      c_encoder_keys.append(c_key)
    # Create array of pointers
    array_type = POINTER(c_uint8) * len(c_encoder_keys)
    c_keys_array = array_type(*c_encoder_keys)    
    c_key_lens, c_values = create_size_array(encoder_key_lens), create_rank_array(encoder_values)

    # Prepare special tokens
    c_special_keys, c_special_values, special_count = None, None, 0
    if special_tokens:
      special_keys_list = list(special_tokens.keys())
      special_values_list = list(special_tokens.values())
      c_special_keys = create_string_array(special_keys_list)
      c_special_values = create_rank_array(special_values_list)
      special_count = len(special_tokens)

    # Initialize the BPE
    pattern_bytes = pattern.encode('utf-8') if isinstance(pattern, str) else pattern
    self._bpe_ptr = self._interface.lib.shred_new(c_keys_array, c_key_lens, c_values, len(encoder_data), c_special_keys, c_special_values, special_count, pattern_bytes)
    if not self._bpe_ptr:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION, "Failed to initialize ShredBPE")
    self._is_initialized = True
  
  def load_from_file(self, vocab_file: Union[str, Path]) -> None:
    """Load vocabulary from a JSON file

    Expected format:
    {
      "encoder": {"base64_bytes": rank, ...},
      "special_tokens": {"token_string": rank, ...},
      "pattern": "regex_pattern"
    }

    Args:
      vocab_file: Path to the vocabulary JSON file
    """
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
      raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
      vocab_data = json.load(f)
    
    # Decode base64 encoded byte sequences
    encoder_data = {}
    for b64_key, rank in vocab_data.get("encoder", {}).items():
      try:
        byte_key = base64.b64decode(b64_key)
        encoder_data[byte_key] = rank
      except Exception:
        # Fallback: treat as UTF-8 string
        encoder_data[b64_key.encode('utf-8')] = rank
    
    special_tokens = vocab_data.get("special_tokens", {})
    pattern = vocab_data.get("pattern", DEFAULT_REGEX)
    self.load_vocab(encoder_data, special_tokens, pattern)
  
  def encode(self, text: str, allowed_special: Optional[List[str]] = None) -> List[int]:
    """Encode text into tokens
    
    Args:
      text: Input text to encode
      allowed_special: List of special tokens that are allowed (None = all allowed)
    
    Returns:
      List of token IDs
    """
    self._check_initialized()    
    # Create token array
    token_array = self._interface.lib.token_array_new(1024)
    if not token_array:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION)

    try:
      text_bytes = text.encode('utf-8')
      if allowed_special is None:
        # Use encode_ordinary for no special tokens
        result = self._interface.lib.encode_ordinary(self._bpe_ptr, text_bytes, token_array)
      else:
        # Use encode with allowed special tokens
        c_allowed = create_string_array(allowed_special) if allowed_special else None
        allowed_count = len(allowed_special) if allowed_special else 0
        result = self._interface.lib.encode(self._bpe_ptr, text_bytes, c_allowed, allowed_count, token_array)
      
      if result != ShredError.OK:
        raise ShredBPEError(result)

      # Extract tokens
      tokens = tokens_from_c_array(token_array.contents.tokens, token_array.contents.count)
      return tokens

    finally:
      self._interface.lib.token_array_free(token_array)
  
  def encode_ordinary(self, text: str) -> List[int]:
    """Encode text without processing special tokens
    
    Args:
      text: Input text to encode
    
    Returns:
      List of token IDs
    """
    return self.encode(text, allowed_special=[])
  
  def encode_bytes(self, data: bytes) -> List[int]:
    """Encode raw bytes into tokens
    
    Args:
      data: Raw bytes to encode
    
    Returns:
      List of token IDs
    """
    self._check_initialized()
    
    token_array = self._interface.lib.token_array_new(1024)
    if not token_array:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION)
    
    try:
      c_bytes, byte_len = create_byte_array_from_bytes(data)
      result = self._interface.lib.encode_bytes(self._bpe_ptr, c_bytes, byte_len, token_array)
      if result != ShredError.OK:
        raise ShredBPEError(result)
      
      tokens = tokens_from_c_array(token_array.contents.tokens, token_array.contents.count)
      return tokens
      
    finally:
      self._interface.lib.token_array_free(token_array)
  
  def encode_with_completions(self, text: str, allowed_special: Optional[List[str]] = None) -> CompletionResult:
    """Encode text and get possible completions for partial tokenization
    
    Args:
      text: Input text to encode
      allowed_special: List of special tokens that are allowed
    
    Returns:
      CompletionResult with tokens and possible completions
    """
    self._check_initialized()
    
    unstable_result = self._interface.lib.encode_unstable_result_new()
    if not unstable_result:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION)

    try:
      text_bytes = text.encode('utf-8')
      c_allowed = create_string_array(allowed_special) if allowed_special else None
      allowed_count = len(allowed_special) if allowed_special else 0
      result = self._interface.lib.encode_with_unstable(self._bpe_ptr, text_bytes, c_allowed, allowed_count, unstable_result)

      if result != ShredError.OK:
        raise ShredBPEError(result)
      
      # Extract main tokens
      main_tokens = tokens_from_c_array(unstable_result.contents.tokens.tokens, unstable_result.contents.tokens.count)
      
      # Extract completions
      completions = []
      completion_set = unstable_result.contents.completions
      for i in range(completion_set.count):
        completion_array = completion_set.completions[i]
        completion_tokens = tokens_from_c_array(completion_array.contents.tokens, completion_array.contents.count)
        completions.append(completion_tokens)

      return CompletionResult(main_tokens, completions)
      
    finally:
      self._interface.lib.encode_unstable_result_free(unstable_result)
  
  def decode(self, tokens: List[int]) -> str:
    """Decode tokens back to text
    
    Args:
      tokens: List of token IDs to decode
    
    Returns:
      Decoded text string
    """
    raw_bytes = self.decode_bytes(tokens)
    try:
      return raw_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
      raise ShredBPEError(ShredError.ERROR_INVALID_UTF8, f"Invalid UTF-8 in decoded bytes: {e}")
  
  def decode_bytes(self, tokens: List[int]) -> bytes:
    """Decode tokens to raw bytes
    
    Args:
      tokens: List of token IDs to decode
    
    Returns:
      Raw decoded bytes
    """
    self._check_initialized()

    byte_array = self._interface.lib.byte_array_new(1024)
    if not byte_array:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION)

    try:
      c_tokens = create_rank_array(tokens)
      result = self._interface.lib.decode_bytes(self._bpe_ptr, c_tokens, len(tokens), byte_array)

      if result != ShredError.OK:
        raise ShredBPEError(result)

      decoded_bytes = bytes_from_c_array(byte_array.contents.bytes, byte_array.contents.len)
      return decoded_bytes

    finally:
      self._interface.lib.byte_array_free(byte_array)
  
  def decode_single_token(self, token: int) -> bytes:
    """Decode a single token to bytes
    
    Args:
      token: Token ID to decode
    
    Returns:
      Raw bytes for the token
    """
    self._check_initialized()
    byte_array = self._interface.lib.byte_array_new(256)
    if not byte_array:
      raise ShredBPEError(ShredError.ERROR_MEMORY_ALLOCATION)

    try:
      result = self._interface.lib.decode_single_token_bytes(self._bpe_ptr, token, byte_array)
      if result != ShredError.OK:
        raise ShredBPEError(result)      
      decoded_bytes = bytes_from_c_array(byte_array.contents.bytes, byte_array.contents.len)
      return decoded_bytes

    finally:
      self._interface.lib.byte_array_free(byte_array)
  
  def encode_single_token(self, piece: Union[str, bytes]) -> int:
    """Encode a single piece to its token ID
    
    Args:
      piece: Text piece or bytes to encode to a single token
    
    Returns:
      Token ID
    """
    self._check_initialized()
    
    if isinstance(piece, str):
      piece_bytes = piece.encode('utf-8')
    else:
      piece_bytes = piece
    c_piece, piece_len = create_byte_array_from_bytes(piece_bytes)
    token_rank = Rank()

    result = self._interface.lib.encode_single_token(self._bpe_ptr, c_piece, piece_len, byref(token_rank))
    if result != ShredError.OK:
      raise ShredBPEError(result)
    return token_rank.value
  
  def get_vocab_size(self) -> int:
    """Get the total number of tokens in the vocabulary
    
    Returns:
      Number of tokens
    """
    self._check_initialized()
    return self._interface.lib.get_token_count(self._bpe_ptr)
  
  def get_all_token_bytes(self) -> List[bytes]:
    """Get byte representation of all tokens in the vocabulary
    
    Returns:
      List of byte sequences for each token
    """
    self._check_initialized()
    results_ptr = POINTER(POINTER(ByteArray))()
    count = c_size_t()
    result = self._interface.lib.get_token_byte_values(self._bpe_ptr, byref(results_ptr), byref(count))

    if result != ShredError.OK:
      raise ShredBPEError(result)
    
    try:
      token_bytes = []
      for i in range(count.value):
        byte_array = results_ptr[i]
        token_byte = bytes_from_c_array(byte_array.contents.bytes, byte_array.contents.len)
        token_bytes.append(token_byte)
      return token_bytes
      
    finally:
      # Free the allocated memory
      for i in range(count.value):
        self._interface.lib.byte_array_free(results_ptr[i])
  
  def set_repo_url(self, repo_url: str) -> None:
    """Update the repository URL for vocabulary loading
    
    Args:
      repo_url: New base URL for vocabulary repository
    """
    self.repo_url = repo_url
  
  def _check_initialized(self):
    """Check if the tokenizer is properly initialized"""
    if not self._is_initialized or not self._bpe_ptr:
      raise ShredBPEError(ShredError.ERROR_NULL_POINTER, "Tokenizer not initialized. Call load_vocab() or load_from_encoding() first.")
  
  def cleanup(self):
    """Clean up resources"""
    if self._bpe_ptr:
      self._interface.lib.shred_free(self._bpe_ptr)
      self._bpe_ptr = None
    self._is_initialized = False
  
  def __del__(self):
    """Destructor to ensure cleanup"""
    self.cleanup()

# Convenience function for quick initialization
def load_encoding(encoding_name: str) -> 'Shred':
  """Convenience function to quickly load a tokenizer with vocab file (no SentencePiece dependency)
  
  Args:
    encoding_name: Name of the encoding (e.g., 'base_50k', 'ava_v1', 'pre_16k')

  Returns:
    Initialized Shred tokenizer instance
  """
  tokenizer = Shred(None, None)
  tokenizer.load_from_encoding(encoding_name)
  return tokenizer
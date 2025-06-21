"""
cbase.py - Low-level ctypes interface for CoreBPE tokenizer

This module provides direct ctypes bindings to the compiled CoreBPE library (.so/.dll).
It defines all the C structures, function signatures, and memory management helpers.
"""

import ctypes
import platform
from ctypes import Structure, POINTER, c_char_p, c_uint32, c_size_t, c_uint8, c_void_p, c_bool, c_int
from typing import Optional, Union
import os

# Type definitions matching the C code
Rank = c_uint32

class ShredError:
  OK = 0
  ERROR_NULL_POINTER = -1
  ERROR_MEMORY_ALLOCATION = -2
  ERROR_INVALID_TOKEN = -3
  ERROR_REGEX_COMPILE = -4
  ERROR_REGEX_MATCH = -5
  ERROR_INVALID_UTF8 = -6

# Structure definitions
class HashMapNode(Structure): pass
class HashMap(Structure): pass
class HashMapStrNode(Structure): pass
class HashMapStr(Structure): pass
class ReverseMapNode(Structure): pass
class ReverseMap(Structure): pass
class SortedTokens(Structure): pass
class TokenArray(Structure): pass
class CompletionSet(Structure): pass
class EncodeUnstableResult(Structure): pass
class ByteArray(Structure): pass
class CoreBPE(Structure): pass

HashMapNode._fields_ = [("key", POINTER(c_uint8)), ("key_len", c_size_t), ("value", Rank), ("next", POINTER(HashMapNode))]
HashMap._fields_ = [("buckets", POINTER(POINTER(HashMapNode))), ("bucket_count", c_size_t), ("size", c_size_t)]
HashMapStrNode._fields_ = [("key", c_char_p), ("value", Rank), ("next", POINTER(HashMapStrNode))]
HashMapStr._fields_ = [("buckets", POINTER(POINTER(HashMapStrNode))), ("bucket_count", c_size_t), ("size", c_size_t)]
ReverseMapNode._fields_ = [("key", Rank), ("value", POINTER(c_uint8)), ("value_len", c_size_t), ("next", POINTER(ReverseMapNode))]
ReverseMap._fields_ = [("buckets", POINTER(POINTER(ReverseMapNode))), ("bucket_count", c_size_t), ("size", c_size_t)]
SortedTokens._fields_ = [("tokens", POINTER(POINTER(c_uint8))), ("token_lens", POINTER(c_size_t)), ("count", c_size_t), ("capacity", c_size_t)]
TokenArray._fields_ = [("tokens", POINTER(Rank)), ("count", c_size_t), ("capacity", c_size_t)]
CompletionSet._fields_ = [("completions", POINTER(POINTER(TokenArray))), ("count", c_size_t), ("capacity", c_size_t)]
EncodeUnstableResult._fields_ = [("tokens", TokenArray), ("completions", CompletionSet)]
ByteArray._fields_ = [("bytes", POINTER(c_uint8)), ("len", c_size_t)]
CoreBPE._fields_ = [("encoder", POINTER(HashMap)), ("special_tokens_encoder", POINTER(HashMapStr)), ("decoder", POINTER(ReverseMap)), ("special_tokens_decoder", POINTER(ReverseMap)), ("regex", c_void_p), ("special_regex", c_void_p), ("sorted_token_bytes", POINTER(SortedTokens))]

class ShredBPEInterface:
  """Low-level ctypes interface to the CoreBPE library"""
  
  def __init__(self, library_path: Optional[str] = None):
    """Initialize the interface with the shared library"""
    if library_path is None:
      # Auto-detect library name based on platform
      if platform.system() == "Windows":
        library_path = "inc/libtoken.dll"
      else:
        library_path = "inc/libtoken.so"
    if not os.path.exists(library_path):
      raise FileNotFoundError(f"CoreBPE library not found at: {library_path}")
    
    self.lib = ctypes.CDLL(library_path)
    self._setup_functions()
  
  def _setup_functions(self):
    """Set up all function signatures and return types"""
    # Core API functions
    self.lib.shred_new.argtypes = [
      POINTER(POINTER(c_uint8)),  # encoder_keys
      POINTER(c_size_t),          # encoder_key_lens
      POINTER(Rank),              # encoder_values
      c_size_t,                   # encoder_count
      POINTER(c_char_p),          # special_token_keys
      POINTER(Rank),              # special_token_values
      c_size_t,                   # special_token_count
      c_char_p                    # pattern
    ]
    self.lib.shred_new.restype = POINTER(CoreBPE)    
    self.lib.shred_free.argtypes = [POINTER(CoreBPE)]
    self.lib.shred_free.restype = None

    # Encoding functions
    self.lib.encode_ordinary.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(TokenArray)]
    self.lib.encode_ordinary.restype = c_int
    self.lib.encode.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(TokenArray)]
    self.lib.encode.restype = c_int
    self.lib.encode_with_unstable.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(EncodeUnstableResult)]
    self.lib.encode_with_unstable.restype = c_int
    self.lib.encode_bytes.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)]
    self.lib.encode_bytes.restype = c_int
    self.lib.encode_single_token.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(Rank)]
    self.lib.encode_single_token.restype = c_int
    self.lib.encode_single_piece.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)]
    self.lib.encode_single_piece.restype = c_int

    # Decoding functions
    self.lib.decode_bytes.argtypes = [POINTER(CoreBPE), POINTER(Rank), c_size_t, POINTER(ByteArray)]
    self.lib.decode_bytes.restype = c_int
    self.lib.decode_single_token_bytes.argtypes = [POINTER(CoreBPE), Rank, POINTER(ByteArray)]
    self.lib.decode_single_token_bytes.restype = c_int

    # Utility functions
    self.lib.get_token_count.argtypes = [POINTER(CoreBPE)]
    self.lib.get_token_count.restype = c_size_t
    self.lib.get_token_byte_values.argtypes = [POINTER(CoreBPE), POINTER(POINTER(ByteArray)), POINTER(c_size_t)]
    self.lib.get_token_byte_values.restype = c_int
    self.lib.token_array_push.argtypes = [POINTER(TokenArray), Rank]
    self.lib.token_array_push.restype = c_int
    
    # Memory management functions
    self.lib.token_array_new.argtypes = [c_size_t]
    self.lib.token_array_new.restype = POINTER(TokenArray)
    self.lib.token_array_free.argtypes = [POINTER(TokenArray)]
    self.lib.token_array_free.restype = None
    self.lib.token_array_clear.argtypes = [POINTER(TokenArray)]
    self.lib.token_array_clear.restype = None
    self.lib.completion_set_new.argtypes = [c_size_t]
    self.lib.completion_set_new.restype = POINTER(CompletionSet)    
    self.lib.completion_set_free.argtypes = [POINTER(CompletionSet)]
    self.lib.completion_set_free.restype = None
    self.lib.encode_unstable_result_new.argtypes = []
    self.lib.encode_unstable_result_new.restype = POINTER(EncodeUnstableResult)
    self.lib.encode_unstable_result_free.argtypes = [POINTER(EncodeUnstableResult)]
    self.lib.encode_unstable_result_free.restype = None
    self.lib.byte_array_new.argtypes = [c_size_t]
    self.lib.byte_array_new.restype = POINTER(ByteArray)
    self.lib.byte_array_free.argtypes = [POINTER(ByteArray)]
    self.lib.byte_array_free.restype = None
    self.lib.byte_array_clear.argtypes = [POINTER(ByteArray)]
    self.lib.byte_array_clear.restype = None
    self.lib.sorted_tokens_new.argtypes = []
    self.lib.sorted_tokens_new.restype = POINTER(SortedTokens)
    self.lib.sorted_tokens_free.argtypes = [POINTER(SortedTokens)]
    self.lib.sorted_tokens_free.restype = None
    self.lib.sorted_tokens_add.argtypes = [POINTER(SortedTokens), POINTER(c_uint8), c_size_t]
    self.lib.sorted_tokens_add.restype = c_int
    self.lib.sorted_tokens_sort.argtypes = [POINTER(SortedTokens)]
    self.lib.sorted_tokens_sort.restype = c_int
    self.lib.sorted_tokens_find_prefix.argtypes = [POINTER(SortedTokens), POINTER(c_uint8), c_size_t]
    self.lib.sorted_tokens_find_prefix.restype = c_size_t
    self.lib.completion_set_add.argtypes = [POINTER(CompletionSet), POINTER(TokenArray)]
    self.lib.completion_set_add.restype = c_int

# Helper functions for memory management and type conversion
def create_byte_array_from_bytes(data: bytes):
  """Convert Python bytes to C uint8_t array"""
  if not data:
    return None, c_size_t(0)
  
  array_type = c_uint8 * len(data)
  c_array = array_type(*data)
  return ctypes.cast(c_array, POINTER(c_uint8)), c_size_t(len(data))

def create_string_array(strings: list[str]):
  """Convert Python string list to C char** array"""
  if not strings:
    return None
  
  array_type = c_char_p * len(strings)
  c_strings = [s.encode('utf-8') if isinstance(s, str) else s for s in strings]
  return array_type(*c_strings)

def create_rank_array(ranks: list[int]):
  """Convert Python int list to C Rank array"""
  if not ranks:
    return None
  
  array_type = Rank * len(ranks)
  return array_type(*ranks)

def create_size_array(sizes: list[int]):
  """Convert Python int list to C size_t array"""
  if not sizes:
    return None
  
  array_type = c_size_t * len(sizes)
  return array_type(*sizes)

def bytes_from_c_array(c_array, length: int) -> bytes:
  """Convert C uint8_t array to Python bytes"""
  if not c_array or length == 0:
    return b''
  
  return bytes(ctypes.cast(c_array, POINTER(c_uint8 * length)).contents)

def tokens_from_c_array(c_array, length: int) -> list[int]:
  """Convert C Rank array to Python int list"""
  if not c_array or length == 0:
    return []
  
  return list(ctypes.cast(c_array, POINTER(Rank * length)).contents)
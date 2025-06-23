import ctypes, os, platform
from ctypes import *

Rank = c_uint32

if platform.system() == 'Windows':
  lib_path = os.path.join(os.path.dirname(__file__), 'lib/libtoken.dll')
  lib = ctypes.CDLL(lib_path, winmode=0)
else:
  lib_path = os.path.join(os.path.dirname(__file__), 'lib/libtoken.so')
  lib = ctypes.CDLL(lib_path)

class ShredError:
  OK, ERROR_NULL_POINTER, ERROR_MEMORY_ALLOCATION = 0, -1, -2
  ERROR_INVALID_TOKEN, ERROR_REGEX_COMPILE, ERROR_REGEX_MATCH, ERROR_INVALID_UTF8 = -3, -4, -5, -6

class TokenArray(Structure): _fields_ = [("tokens", POINTER(Rank)), ("count", c_size_t), ("capacity", c_size_t)]
class CompletionSet(Structure): _fields_ = [("completions", POINTER(POINTER(TokenArray))), ("count", c_size_t), ("capacity", c_size_t)]
class EncodeUnstableResult(Structure): _fields_ = [("tokens", TokenArray), ("completions", CompletionSet)]
class ByteArray(Structure): _fields_ = [("bytes", POINTER(c_uint8)), ("len", c_size_t)]
class CoreBPE(Structure): _fields_ = [("encoder", c_void_p), ("special_tokens_encoder", c_void_p), ("decoder", c_void_p), ("special_tokens_decoder", c_void_p), ("regex", c_void_p), ("special_regex", c_void_p), ("sorted_token_bytes", c_void_p)]

def setup_functions(lib):
  lib.shred_new.argtypes = [POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(Rank), c_size_t, 
                           POINTER(c_char_p), POINTER(Rank), c_size_t, c_char_p]
  lib.shred_new.restype = POINTER(CoreBPE)
  lib.shred_free.argtypes = [POINTER(CoreBPE)]
  lib.shred_free.restype = None

  for func in ['encode_ordinary', 'encode', 'encode_bytes']:
    getattr(lib, func).restype = c_int

  lib.encode_ordinary.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(TokenArray)]
  lib.encode.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(TokenArray)]
  lib.encode_with_unstable.argtypes = [POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(EncodeUnstableResult)]
  lib.encode_with_unstable.restype = c_int
  lib.encode_bytes.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)]
  lib.encode_single_token.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(Rank)]
  lib.encode_single_token.restype = c_int
  lib.encode_single_piece.argtypes = [POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)]
  lib.encode_single_piece.restype = c_int

  lib.decode_bytes.argtypes = [POINTER(CoreBPE), POINTER(Rank), c_size_t, POINTER(ByteArray)]
  lib.decode_bytes.restype = c_int
  lib.decode_single_token_bytes.argtypes = [POINTER(CoreBPE), Rank, POINTER(ByteArray)]
  lib.decode_single_token_bytes.restype = c_int

  lib.get_token_count.argtypes = [POINTER(CoreBPE)]
  lib.get_token_count.restype = c_size_t
  lib.get_token_byte_values.argtypes = [POINTER(CoreBPE), POINTER(POINTER(ByteArray)), POINTER(c_size_t)]
  lib.get_token_byte_values.restype = c_int

  for func in ['token_array_new', 'completion_set_new', 'encode_unstable_result_new', 'byte_array_new']:
    getattr(lib, func).restype = c_void_p
  
  lib.token_array_new.argtypes = [c_size_t]
  lib.completion_set_new.argtypes = [c_size_t]
  lib.byte_array_new.argtypes = [c_size_t]
  
  for func in ['token_array_free', 'completion_set_free', 'encode_unstable_result_free', 'byte_array_free']:
    getattr(lib, func).argtypes = [c_void_p]
    getattr(lib, func).restype = None

  lib.token_array_push.argtypes = [POINTER(TokenArray), Rank]
  lib.token_array_push.restype = c_int

def create_token_array(lib, capacity=1000):
  return ctypes.cast(lib.token_array_new(capacity), POINTER(TokenArray))

def create_byte_array(lib, capacity=1000):
  return ctypes.cast(lib.byte_array_new(capacity), POINTER(ByteArray))

def create_encode_unstable_result(lib):
  return ctypes.cast(lib.encode_unstable_result_new(), POINTER(EncodeUnstableResult))

def check_error(error_code):
  if error_code != ShredError.OK:
    error_msgs = {ShredError.ERROR_NULL_POINTER: "Null pointer", ShredError.ERROR_MEMORY_ALLOCATION: "Memory allocation failed", ShredError.ERROR_INVALID_TOKEN: "Invalid token", 
                  ShredError.ERROR_REGEX_COMPILE: "Regex compilation failed", ShredError.ERROR_REGEX_MATCH: "Regex match failed", ShredError.ERROR_INVALID_UTF8: "Invalid UTF-8"}
    raise RuntimeError(f"CoreBPE error: {error_msgs.get(error_code, f'Unknown error {error_code}')}")

setup_functions(lib)
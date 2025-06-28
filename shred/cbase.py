import ctypes, os, sys, platform, sysconfig
from ctypes import Structure, c_float, c_double, c_int, c_int8, c_int16, c_int32, c_int64, c_uint8, c_uint16, c_uint32, c_uint64, c_size_t, c_void_p, c_char_p, POINTER
from typing import *

def _get_lib_path():
  pkg_dir = os.path.dirname(__file__)
  possible_names = ['token', 'libtoken', 'tokenizer', 'libtokenizer']
  possible_exts = ['.pyd', '.dll', '.so', '.dylib', sysconfig.get_config_var('EXT_SUFFIX') or '']
  search_dirs = [pkg_dir, os.path.join(pkg_dir, 'lib'), os.path.join(pkg_dir, '..', 'build')]
  
  for search_dir in search_dirs:
    if not os.path.exists(search_dir): continue
    for root, dirs, files in os.walk(search_dir):
      for file in files:
        for name in possible_names:
          if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext):
            return os.path.join(root, file)  
  raise FileNotFoundError(f"Could not find tokenizer library in {search_dirs}. Available files: {[f for d in search_dirs if os.path.exists(d) for f in os.listdir(d)]}")

lib = ctypes.CDLL(_get_lib_path(), winmode=0)

class ShredError:
  OK, ERROR_NULL_POINTER, ERROR_MEMORY_ALLOCATION = 0, -1, -2
  ERROR_INVALID_TOKEN, ERROR_REGEX_COMPILE, ERROR_REGEX_MATCH, ERROR_INVALID_UTF8 = -3, -4, -5, -6

class Rank(ctypes.Union): _fields_ = [("value", c_uint32)]
class TokenArray(Structure): _fields_ = [("tokens", POINTER(c_uint32)), ("count", c_size_t), ("capacity", c_size_t)]
class CompletionSet(Structure):  _fields_ = [("completions", POINTER(POINTER(TokenArray))), ("count", c_size_t), ("capacity", c_size_t)]
class EncodeUnstableResult(Structure):  _fields_ = [("tokens", TokenArray), ("completions", CompletionSet)]
class ByteArray(Structure):  _fields_ = [("bytes", POINTER(c_uint8)), ("len", c_size_t)]
class CoreBPE(Structure):  _fields_ = [("encoder", c_void_p), ("special_tokens_encoder", c_void_p), ("decoder", c_void_p), ("special_tokens_decoder", c_void_p), ("regex", c_void_p), ("special_regex", c_void_p), ("sorted_token_bytes", c_void_p)]

def _setup_func(name, argtypes, restype):
  func = getattr(lib, name)
  func.argtypes, func.restype = argtypes, restype
  return func

_funcs = {
  'shred_new': ([POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(c_uint32), c_size_t, POINTER(c_char_p), POINTER(c_uint32), c_size_t, c_char_p], POINTER(CoreBPE)),
  'shred_free': ([POINTER(CoreBPE)], None), 'encode_ordinary': ([POINTER(CoreBPE), c_char_p, POINTER(TokenArray)], c_int),
  'encode': ([POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(TokenArray)], c_int), 'encode_bytes': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)], c_int),
  'encode_single_token': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(c_uint32)], c_int), 'encode_with_unstable': ([POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(EncodeUnstableResult)], c_int),
  'encode_single_piece': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)], c_int), 'decode_bytes': ([POINTER(CoreBPE), POINTER(c_uint32), c_size_t, POINTER(ByteArray)], c_int),
  'decode_single_token_bytes': ([POINTER(CoreBPE), c_uint32, POINTER(ByteArray)], c_int), 'get_token_count': ([POINTER(CoreBPE)], c_size_t),
  'get_token_byte_values': ([POINTER(CoreBPE), POINTER(POINTER(ByteArray)), POINTER(c_size_t)], c_int), 'token_array_new': ([c_size_t], POINTER(TokenArray)),
  'token_array_free': ([POINTER(TokenArray)], None), 'byte_array_new': ([c_size_t], POINTER(ByteArray)),
  'byte_array_free': ([POINTER(ByteArray)], None), 'completion_set_new': ([c_size_t], POINTER(CompletionSet)),
  'completion_set_free': ([POINTER(CompletionSet)], None), 'encode_unstable_result_new': ([], POINTER(EncodeUnstableResult)),
  'encode_unstable_result_free': ([POINTER(EncodeUnstableResult)], None), 'token_array_push': ([POINTER(TokenArray), c_uint32], c_int),
}

for name, (argtypes, restype) in _funcs.items(): _setup_func(name, argtypes, restype)

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
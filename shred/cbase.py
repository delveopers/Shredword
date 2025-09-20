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
          if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext): return os.path.join(root, file)
  raise FileNotFoundError(f"Could not find tokenizer library in {search_dirs}. Available files: {[f for d in search_dirs if os.path.exists(d) for f in os.listdir(d)]}")

lib = ctypes.CDLL(_get_lib_path(), winmode=0)
Rank = c_uint32

class SortedTokens(Structure): _fields_ = [("tokens", POINTER(POINTER(c_uint8))), ("token_lens", POINTER(c_size_t)), ("count", c_size_t), ("capacity", c_size_t)]
class TokenArray(Structure): _fields_ = [("tokens", POINTER(Rank)), ("count", c_size_t), ("capacity", c_size_t)]
class CompletionSet(Structure): _fields_ = [("completions", POINTER(POINTER(TokenArray))), ("count", c_size_t), ("capacity", c_size_t)]
class EncodeUnstableResult(Structure): _fields_ = [("tokens", TokenArray), ("completions", CompletionSet)]
class ByteArray(Structure): _fields_ = [("bytes", POINTER(c_uint8)), ("len", c_size_t)]
class CoreBPE(Structure): _fields_ = [("encoder", c_void_p), ("special_tokens_encoder", c_void_p), ("decoder", c_void_p), ("special_tokens_decoder", c_void_p), ("regex", c_void_p), ("special_regex", c_void_p), ("sorted_token_bytes", c_void_p)]

def _setup_func(name, argtypes, restype):
  func = getattr(lib, name)
  func.argtypes, func.restype = argtypes, restype
  return func

_funcs = {
  'shredCreate': ([POINTER(POINTER(c_uint8)), POINTER(c_size_t), POINTER(Rank), c_size_t, POINTER(c_char_p), POINTER(Rank), c_size_t, c_char_p], POINTER(CoreBPE)), 'shredFree': ([POINTER(CoreBPE)], None),
  'encodeOrdinary': ([POINTER(CoreBPE), c_char_p, POINTER(TokenArray)], None), 'encode': ([POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(TokenArray)], None),
  'encodeWithUnstable': ([POINTER(CoreBPE), c_char_p, POINTER(c_char_p), c_size_t, POINTER(EncodeUnstableResult)], None), 'encodeBytes': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)], None),
  'encodeSingleToken': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(Rank)], None), 'encodeSinglePiece': ([POINTER(CoreBPE), POINTER(c_uint8), c_size_t, POINTER(TokenArray)], None),
  'decodeBytes': ([POINTER(CoreBPE), POINTER(Rank), c_size_t, POINTER(ByteArray)], None), 'decodeSingleTokenBytes': ([POINTER(CoreBPE), Rank, POINTER(ByteArray)], None),
  'getTokenCount': ([POINTER(CoreBPE)], c_size_t), 'getTokenByteValues': ([POINTER(CoreBPE), POINTER(POINTER(ByteArray)), POINTER(c_size_t)], None),
  'tokenArrayCreate': ([c_size_t], POINTER(TokenArray)), 'tokenArrayFree': ([POINTER(TokenArray)], None), 'tokenArrayClear': ([POINTER(TokenArray)], None), 'tokenArrayPush': ([POINTER(TokenArray), Rank], None),
  'completionSetCreate': ([c_size_t], POINTER(CompletionSet)), 'completionSetFree': ([POINTER(CompletionSet)], None), 'completionSetAdd': ([POINTER(CompletionSet), POINTER(TokenArray)], None), 'encodeUnstableResultCreate': ([], POINTER(EncodeUnstableResult)), 'encodeUnstableResultFree': ([POINTER(EncodeUnstableResult)], None),
  'byteArrayCreate': ([c_size_t], POINTER(ByteArray)), 'byteArrayFree': ([POINTER(ByteArray)], None), 'byteArrayClear': ([POINTER(ByteArray)], None),
  'sortedTokensCreate': ([], POINTER(SortedTokens)), 'sortedTokensFree': ([POINTER(SortedTokens)], None), 'sortedTokensAdd': ([POINTER(SortedTokens), POINTER(c_uint8), c_size_t], None), 'sortedTokensSort': ([POINTER(SortedTokens)], None), 'sortedTokensFindPrefix': ([POINTER(SortedTokens), POINTER(c_uint8), c_size_t], c_size_t),
}

for name, (argtypes, restype) in _funcs.items(): _setup_func(name, argtypes, restype)

def create_token_array(lib, capacity=1000): return lib.tokenArrayCreate(capacity)
def create_byte_array(lib, capacity=1000): return lib.byteArrayCreate(capacity)
def create_encode_unstable_result(lib): return lib.encodeUnstableResultCreate()
def create_completion_set(lib, capacity=10): return lib.completionSetCreate(capacity)
def create_sorted_tokens(lib): return lib.sortedTokensCreate()
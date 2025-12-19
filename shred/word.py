import os
from typing import List, Optional, Tuple, Dict
from .cbase import lib, c_char_p, c_int, POINTER, c_float
from ctypes import byref, string_at
import numpy as np

class WordTokenizer:
  def __init__(self):
    self._tokens_struct = None

  def tokenize(self, text: str) -> List[str]:
    if not text: return []
    text_bytes = text.encode('utf-8')
    self._tokens_struct = lib.tokenize(text_bytes)
    if not self._tokens_struct.items: return []
    return [string_at(self._tokens_struct.items[i]).decode('utf-8', errors='replace') for i in range(self._tokens_struct.count)]

  def __del__(self):
    if self._tokens_struct and self._tokens_struct.items:
      lib.free_tokens(byref(self._tokens_struct))
      self._tokens_struct = None

class TfidfEmbedding:
  def __init__(self):
    self.vocab = lib.vocab_create()
    if not self.vocab: raise RuntimeError("Failed to create vocabulary")

  def add_document(self, text: str):
    if not text: return
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text)
    if not tokens: return
    tokens_array = (c_char_p * len(tokens))(*[t.encode('utf-8') for t in tokens])
    lib.vocab_add_document(self.vocab, tokens_array, len(tokens))

  def add_documents(self, documents: List[str]):
    for doc in documents: self.add_document(doc)

  def encode_ids(self, text: str) -> List[int]:
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text)
    if not tokens: return []
    tokens_array = (c_char_p * len(tokens))(*[t.encode('utf-8') for t in tokens])
    out_n = c_int()
    ids_ptr = lib.encode_ids(self.vocab, tokens_array, len(tokens), byref(out_n))
    if not ids_ptr: return []
    try: return [ids_ptr[i] for i in range(out_n.value)]
    finally: lib.free_buffer(ids_ptr)

  def encode_tfidf_dense(self, text: str) -> np.ndarray:
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text)
    if not tokens: return np.array([], dtype=np.float32)
    tokens_array = (c_char_p * len(tokens))(*[t.encode('utf-8') for t in tokens])
    vec_ptr = lib.encode_tfidf_dense(self.vocab, tokens_array, len(tokens))
    if not vec_ptr: return np.array([], dtype=np.float32)
    try:
      vocab_size = self.vocab.contents.size
      return np.ctypeslib.as_array(vec_ptr, shape=(vocab_size,)).copy()
    finally: lib.free_buffer(vec_ptr)

  def encode_tfidf_sparse(self, text: str) -> Tuple[List[int], List[float]]:
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(text)
    if not tokens: return [], []
    tokens_array = (c_char_p * len(tokens))(*[t.encode('utf-8') for t in tokens])
    indices_ptr, values_ptr, out_n = POINTER(c_int)(), POINTER(c_float)(), c_int()
    lib.encode_tfidf_sparse(self.vocab, tokens_array, len(tokens), byref(indices_ptr), byref(values_ptr), byref(out_n))
    if not indices_ptr or not values_ptr or out_n.value == 0: return [], []
    try:
      indices = [indices_ptr[i] for i in range(out_n.value)]
      values = [values_ptr[i] for i in range(out_n.value)]
      return indices, values
    finally:
      lib.free_buffer(indices_ptr)
      lib.free_buffer(values_ptr)

  def save(self, path: str) -> Optional[str]:
    path_bytes = path.encode('utf-8')
    err_ptr = lib.vocab_save(self.vocab, path_bytes)
    if err_ptr:
      err_msg = string_at(err_ptr).decode('utf-8', errors='replace')
      lib.free_buffer(err_ptr)
      return err_msg
    return None

  @staticmethod
  def load(path: str) -> 'TfidfEmbedding':
    if not os.path.exists(path): raise FileNotFoundError(f"Vocabulary file not found: {path}")
    embedding = TfidfEmbedding.__new__(TfidfEmbedding)
    path_bytes = path.encode('utf-8')
    vocab_ptr = POINTER(type(lib.vocab_create().contents))()
    err_ptr = lib.vocab_load(byref(vocab_ptr), path_bytes)
    if err_ptr:
      err_msg = string_at(err_ptr).decode('utf-8', errors='replace')
      lib.free_buffer(err_ptr)
      raise RuntimeError(f"Failed to load vocabulary: {err_msg}")
    embedding.vocab = vocab_ptr
    return embedding

  @property
  def vocab_size(self) -> int: return self.vocab.contents.size if self.vocab else 0
  @property
  def document_count(self) -> int: return self.vocab.contents.docs if self.vocab else 0
  
  def get_vocabulary(self) -> Dict[str, Tuple[int, int]]:
    if not self.vocab: return {}
    v = self.vocab.contents
    return {string_at(v.words[i]).decode('utf-8', errors='replace'): (v.counts[i], v.df[i]) for i in range(v.size)}

  def lookup(self, token: str) -> int:
    if not self.vocab: return -1
    token_bytes = token.encode('utf-8')
    return lib.vocab_lookup(self.vocab, token_bytes)

  def __del__(self):
    if hasattr(self, 'vocab') and self.vocab: lib.vocab_free(self.vocab)
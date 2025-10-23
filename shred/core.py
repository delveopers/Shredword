# --- patched shred front-end (tokenizer.py) ---
import json, re, os
from typing import List, Dict, Optional, Sequence
from .cbase import lib, create_token_array, create_byte_array, create_encode_unstable_result
from ctypes import POINTER, c_uint8, c_size_t, c_uint32, c_char_p, create_string_buffer, cast, byref, string_at

BASIC_REGEX = r"'s|'t|'re|'ve|'d|'ll|'m|[A-Za-z]+|\d+|\r?\n|\s+|[^\w\s]"

def _get_vocab_path(encoding_name: str) -> str:
  pkg_dir = os.path.dirname(__file__)
  vocab_dirs = [os.path.join(pkg_dir, 'vocabs'), os.path.join(pkg_dir, '..', 'vocabs'), os.path.join(pkg_dir, '..', '..', 'vocabs')]
  for vocab_dir in vocab_dirs:
    if os.path.exists(vocab_dir):
      vocab_file = os.path.join(vocab_dir, f'{encoding_name}.model')
      if os.path.exists(vocab_file): return vocab_file
  raise FileNotFoundError(f"Vocab file '{encoding_name}.model' not found in any of: {vocab_dirs}")


class Shred:
  def __init__(self):
    self.bpe, self._vocab, self._special_tokens, self._encoder, self._decoder = None, [], {}, {}, {}
    # keeping buffers and supporting ctypes arrays as attributes to preserve lifetime
    self._encoder_buffers, self._encoder_keys, self._encoder_key_lens, self._encoder_values, self._special_keys, self._special_values, self._pattern_buf = [], None, None, None, None, None, None
    self._single_byte_encoder, self._pattern, self._pattern_re = {}, BASIC_REGEX, re.compile(BASIC_REGEX)
    self._special_token_bytes: Dict[str, bytes] = {}

  def load_from_encoding(self, encoding_name: str, local: bool = True):
    if local:
      vocab_path = _get_vocab_path(encoding_name)
      with open(vocab_path, 'rb') as f: vocab_data = self._parse_model_file(f.read(), encoding_name)
    else: vocab_data = self._download_vocab(encoding_name)
    self._vocab, self._special_tokens = vocab_data['vocab'], vocab_data.get('special_tokens', {})
    pattern = vocab_data.get('pattern', self._pattern)
    if pattern != self._pattern: self._pattern, self._pattern_re = pattern, re.compile(pattern)
    # precompute special token bytes map
    self._special_token_bytes = {tok: tok.encode('utf-8') for tok in self._special_tokens.keys()}
    self._build_mappings()
    self._initialize_bpe(self._pattern)

  def _download_vocab(self, encoding_name: str) -> Dict:
    try:
      import urllib.request
      base_urls = [f"https://raw.githubusercontent.com/delveopers/shredword/{branch}/shred/vocabs/{encoding_name}.model" for branch in ["main", "dev"]]
      last_exc = None
      for url in base_urls:
        try:
          with urllib.request.urlopen(url) as response: return self._parse_model_file(response.read(), encoding_name)
        except Exception as e: last_exc = e; continue
      raise ValueError(f"Failed to load encoding '{encoding_name}' from any source: {last_exc}")
    except ImportError:
      raise RuntimeError("urllib not available and local vocab not found. Install package properly or provide vocab file.")

  def _build_mappings(self):
    encoder, decoder, single_byte = {}, {}, {}
    for i, token in enumerate(self._vocab):
      if not token: continue
      if token.startswith('<0x') and token.endswith('>') and len(token) == 6:
        try: byte_val = int(token[3:5], 16)
        except ValueError: continue
        token_bytes = bytes([byte_val])
        encoder[token_bytes], decoder[i], single_byte[byte_val] = i, token_bytes, i
      elif not (token.startswith('<') and token.endswith('>')):
        try: token_bytes = token.encode('utf-8')
        except UnicodeEncodeError: continue
        if token_bytes:
          encoder[token_bytes], decoder[i] = i, token_bytes
          if len(token_bytes) == 1: single_byte[token_bytes[0]] = i
    self._encoder, self._decoder, self._single_byte_encoder = encoder, decoder, single_byte

  def _parse_model_file(self, content: bytes, encoding_name: str) -> Dict:
    try:
      vocab_dict = json.loads(content.decode('utf-8'))
      max_rank = max(vocab_dict.values()) if vocab_dict else 0
      vocab_list, special_tokens = [''] * (max_rank + 1), {}
      for token_str, rank in vocab_dict.items():
        clean_token = token_str.strip('"\'')
        if 0 <= rank <= max_rank: vocab_list[rank] = clean_token
        if clean_token.startswith('<') and clean_token.endswith('>') and not clean_token.startswith('<0x'): special_tokens[clean_token] = rank
      return {'vocab': vocab_list, 'special_tokens': special_tokens, 'pattern': BASIC_REGEX}
    except Exception as e: raise ValueError(f"Unable to parse model file for encoding '{encoding_name}': {e}")

  def _initialize_bpe(self, pattern: str):
    if not self._encoder: raise RuntimeError("Encoder not built")
    sorted_items = sorted(self._encoder.items(), key=lambda x: x[1])
    self._encoder_buffers, n = [], len(sorted_items)
    EncoderKeyArrayType, EncoderLensArrayType, EncoderValuesArrayType = POINTER(c_uint8) * n, c_size_t * n, c_uint32 * n
    encoder_keys, encoder_key_lens, encoder_values = EncoderKeyArrayType(), EncoderLensArrayType(), EncoderValuesArrayType()
    for i, (token_bytes, rank) in enumerate(sorted_items):
      buf = create_string_buffer(token_bytes) # create_string_buffer keeps a stable buffer; store it so it lives for lifetime of tokenizer
      self._encoder_buffers.append(buf)
      encoder_keys[i] = cast(buf, POINTER(c_uint8))       # cast the buffer to POINTER(c_uint8)
      encoder_key_lens[i] = len(token_bytes)
      encoder_values[i] = rank
    self._encoder_keys, self._encoder_key_lens, self._encoder_values = encoder_keys, encoder_key_lens, encoder_values     # keep references to arrays on self to preserve lifetime

    special_count = len(self._special_tokens)
    if special_count > 0:
      SpecialKeysArray, SpecialValuesArray = c_char_p * special_count, c_uint32 * special_count
      special_keys, special_values = SpecialKeysArray(), SpecialValuesArray()
      # useing the precomputed bytes for special tokens
      for i, (token, rank) in enumerate(self._special_tokens.items()):
        special_keys[i] = self._special_token_bytes.get(token, token.encode('utf-8')) # token is str, but we have precomputed bytes
        special_values[i] = rank
      self._special_keys, self._special_values = special_keys, special_values
    else:
      self._special_keys, self._special_values = None, None
    pattern_buf = create_string_buffer(pattern.encode('utf-8'))     # pattern buffer — cast to c_char_p and keep it alive on self
    self._pattern_buf = pattern_buf
    pattern_c = cast(pattern_buf, c_char_p)
    self.bpe = lib.shredCreate(self._encoder_keys, self._encoder_key_lens, self._encoder_values, n, self._special_keys, self._special_values, special_count, pattern_c)
    if not self.bpe: raise RuntimeError("shredCreate returned NULL")

  def encode(self, text: str, allowed_special: Optional[Sequence[str]] = None) -> List[int]:
    """
    If allowed_special is None -> use encode_ordinary (fast path)
    If allowed_special is [] -> explicitly allow zero special tokens (pass None/0 to native)
    If allowed_special == "all" -> include all known special tokens
    """
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    if allowed_special is None: return self.encode_ordinary(text)     # handle explicit None differently from empty list
    if allowed_special == "all": allowed_special_list = list(self._special_tokens.keys())     # normalize allowed_special (string -> list, "all" -> all)
    elif isinstance(allowed_special, str): allowed_special_list = [allowed_special]
    else: allowed_special_list = list(allowed_special)
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")
    try:
      text_bytes = text.encode('utf-8')
      # if allowed_special_list is empty, pass NULL and 0 (native likely handles that)
      if len(allowed_special_list) == 0: lib.encode(self.bpe, text_bytes, None, 0, token_array)
      else:
        # use precomputed bytes for special tokens to avoid repeated .encode(...)
        special_c_bytes = [self._special_token_bytes.get(s, s.encode('utf-8')) for s in allowed_special_list]
        special_array = (c_char_p * len(special_c_bytes))(*special_c_bytes)
        lib.encode(self.bpe, text_bytes, special_array, len(special_c_bytes), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    except Exception: return self._fallback_encode(text, allowed_special_list)
    finally: lib.tokenArrayFree(token_array)

  def encode_ordinary(self, text: str) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")
    try:
      lib.encodeOrdinary(self.bpe, text.encode('utf-8'), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    except Exception: return self._fallback_encode(text, None)
    finally: lib.tokenArrayFree(token_array)

  def _fallback_encode(self, text: str, allowed_special: Optional[Sequence[str]]) -> List[int]:
    tokens, encoder, single_byte, stokens = [], self._encoder, self._single_byte_encoder, self._special_tokens
    if allowed_special:
      # create a regex group for the allowed special tokens
      special_pattern = '|'.join(re.escape(token) for token in allowed_special)
      parts = re.split(f'({special_pattern})', text) if special_pattern else [text]
      for part in parts:
        if not part: continue
        rank = stokens.get(part)
        if rank is not None: tokens.append(rank)
        else:
          for piece in self._pattern_re.findall(part):
            try: piece_bytes = piece.encode('utf-8')
            except Exception: continue
            rank = encoder.get(piece_bytes)
            if rank is not None: tokens.append(rank)
            else:
              for b in piece_bytes:
                sb_rank = single_byte.get(b)
                tokens.append(sb_rank if sb_rank is not None else 0)
    else:
      for piece in self._pattern_re.findall(text):
        try: piece_bytes = piece.encode('utf-8')
        except Exception: continue
        rank = encoder.get(piece_bytes)
        if rank is not None: tokens.append(rank)
        else:
          for b in piece_bytes:
            sb_rank = single_byte.get(b)
            tokens.append(sb_rank if sb_rank is not None else 0)
    return tokens

  def decode(self, tokens: List[int]) -> str:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    if not tokens: return ""
    byte_array = create_byte_array(lib)
    if not byte_array: raise RuntimeError("Failed to create byte array")
    try:
      tokens_array = (c_uint32 * len(tokens))(*tokens)
      lib.decodeBytes(self.bpe, tokens_array, len(tokens), byte_array)
      out_len = byte_array.contents.len
      if out_len == 0: return ""
      return string_at(byte_array.contents.bytes, out_len).decode('utf-8', errors='replace')
    finally: lib.byteArrayFree(byte_array)

  def decode_single_token(self, token: int) -> bytes:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    byte_array = create_byte_array(lib)
    if not byte_array: raise RuntimeError("Failed to create byte array")
    try:
      lib.decodeSingleTokenBytes(self.bpe, token, byte_array)
      out_len = byte_array.contents.len
      return b"" if out_len == 0 else string_at(byte_array.contents.bytes, out_len)
    finally: lib.byteArrayFree(byte_array)

  def encode_unstable(self, text: str, allowed_special: Optional[Sequence[str]] = None):
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    result = create_encode_unstable_result(lib)
    if not result: raise RuntimeError("Failed to allocate encode_unstable result")
    try:
      text_b = text.encode('utf-8')
      if allowed_special:
        if allowed_special == "all": allowed_special_list = list(self._special_tokens.keys())
        elif isinstance(allowed_special, str): allowed_special_list = [allowed_special]
        else: allowed_special_list = list(allowed_special)
        if len(allowed_special_list) == 0: lib.encodeWithUnstable(self.bpe, text_b, None, 0, result)
        else:
          special_c_bytes = [self._special_token_bytes.get(s, s.encode('utf-8')) for s in allowed_special_list]
          special_array = (c_char_p * len(special_c_bytes))(*special_c_bytes)
          lib.encodeWithUnstable(self.bpe, text_b, special_array, len(special_c_bytes), result)
      else: lib.encodeWithUnstable(self.bpe, text_b, None, 0, result)

      # cache locals to reduce attribute lookup overhead
      res = result.contents
      tokens_count = res.tokens.count
      tokens, completions = [res.tokens.tokens[i] for i in range(tokens_count)], []
      comp_set = res.completions
      for i in range(comp_set.count):
        token_arr_ptr = comp_set.completions[i].contents
        ccount = token_arr_ptr.count
        completions.append([token_arr_ptr.tokens[j] for j in range(ccount)])
      return {'tokens': tokens, 'completions': completions}
    finally: lib.encodeUnstableResultFree(result)

  def encode_bytes(self, data: bytes) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")
    try:
      # keep it simple and copy into a temporary c_uint8 array; it's safe and portable
      data_ptr = (c_uint8 * len(data))(*data)
      lib.encodeBytes(self.bpe, data_ptr, len(data), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    finally: lib.tokenArrayFree(token_array)

  def encode_single_token(self, piece: bytes) -> Optional[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    try:
      piece_ptr, result = (c_uint8 * len(piece))(*piece), c_uint32()
      lib.encodeSingleToken(self.bpe, piece_ptr, len(piece), byref(result))
      return int(result.value)
    except Exception: return None

  def encode_single_piece(self, piece: bytes) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")
    try:
      piece_ptr = (c_uint8 * len(piece))(*piece)
      lib.encodeSinglePiece(self.bpe, piece_ptr, len(piece), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    finally: lib.tokenArrayFree(token_array)

  @property
  def vocab_size(self) -> int: return lib.getTokenCount(self.bpe) if self.bpe else len(self._vocab)
  @property
  def special_tokens(self) -> Dict[str, int]: return self._special_tokens.copy()
  @property
  def vocab(self) -> List[str]: return self._vocab.copy()
  @property
  def encoder(self) -> Dict[bytes, int]: return self._encoder.copy()
  @property
  def decoder(self) -> Dict[int, bytes]: return self._decoder.copy()

  def __del__(self):
    if self.bpe: lib.shredFree(self.bpe)

def load_encoding(encoding_name: str, local: bool = True) -> Shred:
  tokenizer = Shred()
  tokenizer.load_from_encoding(encoding_name, local)
  return tokenizer
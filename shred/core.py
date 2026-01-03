import json, re, os
from typing import List, Dict, Optional, Sequence
from ctypes import POINTER, c_uint8, c_size_t, c_uint32, c_char_p, create_string_buffer, cast, byref, string_at
from .cbase import lib, create_token_array, create_byte_array, create_encode_unstable_result

BASIC_REGEX = r"'s|'t|'re|'ve|'d|'ll|'m|[A-Za-z]+|\d+|\r?\n|\s+|[^\w\s]"

def _get_vocab_path(encoding_name: str) -> str:
  pkg_dir = os.path.dirname(__file__)
  vocab_dirs = [os.path.join(pkg_dir, 'vocabs'), os.path.join(pkg_dir, '..', 'vocabs'), os.path.join(pkg_dir, '..', '..', 'vocabs')]
  for vocab_dir in vocab_dirs:
    if os.path.exists(vocab_dir):
      vocab_file = os.path.join(vocab_dir, f'{encoding_name}.model')
      if os.path.exists(vocab_file):
        return vocab_file
  raise FileNotFoundError(f"Vocab file '{encoding_name}.model' not found")

class Shred:
  def __init__(self):
    self.bpe = None
    self._vocab, self._special_tokens, self._encoder, self._decoder, self._single_byte_encoder = [], {}, {}, {}, {}
    self._pattern, self._pattern_re, self._special_token_bytes = BASIC_REGEX, re.compile(BASIC_REGEX), {}

    self._encoder_buffers, self._encoder_keys, self._encoder_key_lens = [], None, None
    self._encoder_values, self._special_keys, self._special_values, self._pattern_buf = None, None, None, None
    self._token_array, self._byte_array, self._unstable_result, self._all_special_array, self._all_special_count = None, None, None, None, 0

  def load_from_encoding(self, encoding_name: str, local: bool = True):
    if local:
      path = _get_vocab_path(encoding_name)
      with open(path, "rb") as f: data = self._parse_model_file(f.read())
    else: raise RuntimeError("Remote vocab loading disabled")

    self._vocab, self._special_tokens, self._pattern = data["vocab"], data["special_tokens"], data.get("pattern", BASIC_REGEX)
    self._pattern_re = re.compile(self._pattern)
    self._special_token_bytes = {k: k.encode("utf-8") for k in self._special_tokens}
    self._build_mappings()
    self._initialize_bpe()

  def _parse_model_file(self, content: bytes) -> Dict:
    vocab_dict = json.loads(content.decode("utf-8"))
    max_rank = max(vocab_dict.values(), default=0)
    vocab, special = [""] * (max_rank + 1), {}
    for token, rank in vocab_dict.items():
      token = token.strip('"\'')
      vocab[rank] = token
      if token.startswith("<") and token.endswith(">") and not token.startswith("<0x"): special[token] = rank
    return {"vocab": vocab, "special_tokens": special, "pattern": BASIC_REGEX}

  def _build_mappings(self):
    encoder, decoder, single = {}, {}, {}
    for i, token in enumerate(self._vocab):
      if not token: continue
      if token.startswith("<0x") and token.endswith(">") and len(token) == 6:
        try: b = int(token[3:5], 16)
        except ValueError: continue
        encoder[bytes([b])] = i
        decoder[i] = bytes([b])
        single[b] = i
      elif not (token.startswith("<") and token.endswith(">")):
        try: b = token.encode("utf-8")
        except Exception: continue
        encoder[b] = i
        decoder[i] = b
        if len(b) == 1: single[b[0]] = i
    self._encoder, self._decoder, self._single_byte_encoder = encoder, decoder, single

  def _initialize_bpe(self):
    items = sorted(self._encoder.items(), key=lambda x: x[1])
    n = len(items)

    KeyArr, LenArr, ValArr = POINTER(c_uint8) * n, c_size_t * n, c_uint32 * n
    self._encoder_buffers.clear()
    keys, lens, vals = KeyArr(), LenArr(), ValArr()
    for i, (b, r) in enumerate(items):
      buf = create_string_buffer(b)
      self._encoder_buffers.append(buf)
      keys[i], lens[i], vals[i] = cast(buf, POINTER(c_uint8)), len(b), r
    self._encoder_keys, self._encoder_key_lens, self._encoder_values = keys, lens, vals

    sc = len(self._special_tokens)
    if sc:
      SK, SV = c_char_p * sc, c_uint32 * sc
      sk, sv = SK(), SV()
      for i, (tok, rank) in enumerate(self._special_tokens.items()):
        sk[i] = self._special_token_bytes[tok]
        sv[i] = rank
      self._special_keys, self._special_values, self._all_special_array, self._all_special_count = sk, sv, sk, sc
    else: self._special_keys, self._special_values = None, None
    self._pattern_buf = create_string_buffer(self._pattern.encode("utf-8"))
    self.bpe = lib.shredCreate(self._encoder_keys, self._encoder_key_lens, self._encoder_values, n, self._special_keys, self._special_values, sc, cast(self._pattern_buf, c_char_p))
    if not self.bpe: raise RuntimeError("Failed to initialize tokenizer")
    self._token_array = create_token_array(lib)
    self._byte_array = create_byte_array(lib)
    self._unstable_result = create_encode_unstable_result(lib)

  def encode(self, text: str, allowed_special: Optional[Sequence[str]] = None) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    lib.tokenArrayClear(self._token_array)
    data = text.encode("utf-8")
    if allowed_special is None: lib.encodeOrdinary(self.bpe, data, self._token_array)
    else:
      if allowed_special == "all":
        lib.encode(self.bpe, data, self._all_special_array, self._all_special_count, self._token_array)
      elif not allowed_special:
        lib.encode(self.bpe, data, None, 0, self._token_array)
      else:
        arr = (c_char_p * len(allowed_special))(*[self._special_token_bytes[s] for s in allowed_special])
        lib.encode(self.bpe, data, arr, len(allowed_special), self._token_array)
    a = self._token_array.contents
    return [a.tokens[i] for i in range(a.count)]

  def encode_ordinary(self, text: str) -> List[int]: return self.encode(text, None)

  def decode(self, tokens: List[int]) -> str:
    if not tokens: return ""
    lib.byteArrayClear(self._byte_array)
    arr = (c_uint32 * len(tokens))(*tokens)
    lib.decodeBytes(self.bpe, arr, len(tokens), self._byte_array)
    b = self._byte_array.contents
    return string_at(b.bytes, b.len).decode("utf-8", errors="replace")

  def encode_bytes(self, data: bytes) -> List[int]:
    lib.tokenArrayClear(self._token_array)
    buf = (c_uint8 * len(data))(*data)
    lib.encodeBytes(self.bpe, buf, len(data), self._token_array)
    a = self._token_array.contents
    return [a.tokens[i] for i in range(a.count)]

  def encode_single_token(self, piece: bytes) -> Optional[int]:
    out = c_uint32()
    buf = (c_uint8 * len(piece))(*piece)
    lib.encodeSingleToken(self.bpe, buf, len(piece), byref(out))
    return int(out.value)

  def encode_single_piece(self, piece: bytes) -> List[int]:
    lib.tokenArrayClear(self._token_array)
    buf = (c_uint8 * len(piece))(*piece)
    lib.encodeSinglePiece(self.bpe, buf, len(piece), self._token_array)
    a = self._token_array.contents
    return [a.tokens[i] for i in range(a.count)]

  def decode_single_token(self, token: int) -> bytes:
    lib.byteArrayClear(self._byte_array)
    lib.decodeSingleTokenBytes(self.bpe, token, self._byte_array)
    b = self._byte_array.contents
    return b"" if b.len == 0 else string_at(b.bytes, b.len)

  def encode_unstable(self, text: str, allowed_special: Optional[Sequence[str]] = None):
    data = text.encode("utf-8")
    if allowed_special == "all": lib.encodeWithUnstable(self.bpe, data, self._all_special_array, self._all_special_count, self._unstable_result)
    else: lib.encodeWithUnstable(self.bpe, data, None, 0, self._unstable_result)

    r = self._unstable_result.contents
    tokens = [r.tokens.tokens[i] for i in range(r.tokens.count)]

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
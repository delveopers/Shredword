import urllib.request, re, json
from typing import List, Dict, Optional, Union
from .cbase import lib, create_token_array, create_byte_array, create_encode_unstable_result
from ctypes import *

# BASIC_REGEX = r"'(?:s|t|re|ve|d|ll|m)|[^ \r\nA-Za-z0-9]?[A-Za-z]+|[^ \r\nA-Za-z0-9]?[0-9]+|\s*[\r\n]|\s+|[^A-Za-z0-9\s]"
BASIC_REGEX = r"'s|'t|'re|'ve|'d|'ll|'m|[A-Za-z]+|\d+|\r?\n|\s+|[^\w\s]"

class Shred:
  def __init__(self):
    self.bpe, self._vocab, self._special_tokens = None, [], {}
    self._encoder, self._decoder, self._encoder_buffers = {}, {}, []

  def load_from_encoding(self, encoding_name: str, download: bool = True):
    if download:
      vocab_data = self._download_vocab(encoding_name)
    else:
      with open(encoding_name, "r", encoding="utf-8") as f:
        content = f.read()
      vocab_data = self._parse_model_file(content.encode("utf-8"), encoding_name)

    self._vocab, self._special_tokens = vocab_data['vocab'], vocab_data.get('special_tokens', {})
    self._build_mappings()
    self._initialize_bpe(vocab_data.get('pattern', BASIC_REGEX))

  def _download_vocab(self, encoding_name: str) -> Dict:
    base_urls = [
      f"https://raw.githubusercontent.com/delveopers/shredword/main/vocabs/{encoding_name}.model",
      f"https://raw.githubusercontent.com/delveopers/shredword/dev/vocabs/{encoding_name}.model"
    ]

    for url in base_urls:
      try:
        with urllib.request.urlopen(url) as response: return self._parse_model_file(response.read(), encoding_name)
      except: continue
    raise ValueError(f"Failed to load encoding '{encoding_name}' from any source")

  def _build_mappings(self):
    self._encoder, self._decoder = {}, {}
    for i, token in enumerate(self._vocab):
      if not token: continue
      if token.startswith('<0x') and token.endswith('>') and len(token) == 6:
        try:
          byte_val = int(token[3:5], 16)
          token_bytes = bytes([byte_val])
          self._encoder[token_bytes] = i
          self._decoder[i] = token_bytes
        except ValueError: continue
      elif token.startswith('<') and token.endswith('>'): continue
      else:
        try:
          token_bytes = token.encode('utf-8')
          if len(token_bytes) > 0:
            self._encoder[token_bytes] = i
            self._decoder[i] = token_bytes
        except UnicodeEncodeError: continue

  def _parse_model_file(self, content: bytes, encoding_name: str) -> Dict:
    try:
      text_content = content.decode('utf-8')
      vocab_dict = json.loads(text_content)
      max_rank = max(vocab_dict.values()) if vocab_dict else 0
      vocab_list = [''] * (max_rank + 1)
      for token_str, rank in vocab_dict.items():
        clean_token = token_str.strip('"\'')
        if 0 <= rank <= max_rank: vocab_list[rank] = clean_token

      special_tokens = {}
      for token_str, rank in vocab_dict.items():
        clean_token = token_str.strip('"\'')
        if clean_token.startswith('<') and clean_token.endswith('>') and not clean_token.startswith('<0x'): special_tokens[clean_token] = rank
      return {'vocab': vocab_list, 'special_tokens': special_tokens, 'pattern': BASIC_REGEX}
    except Exception as e: raise ValueError(f"Unable to parse model file for encoding '{encoding_name}': {e}")

  def _initialize_bpe(self, pattern: str):
    if not self._encoder: raise RuntimeError("Encoder not built")

    sorted_items = sorted(self._encoder.items(), key=lambda x: x[1])
    self._encoder_buffers = []

    encoder_keys = (POINTER(c_uint8) * len(sorted_items))()
    encoder_key_lens = (c_size_t * len(sorted_items))()
    encoder_values = (c_uint32 * len(sorted_items))()

    for i, (token_bytes, rank) in enumerate(sorted_items):
      buffer = create_string_buffer(token_bytes)
      self._encoder_buffers.append(buffer)
      encoder_keys[i] = cast(buffer, POINTER(c_uint8))
      encoder_key_lens[i] = len(token_bytes)
      encoder_values[i] = rank

    special_count = len(self._special_tokens)
    if special_count > 0:
      special_keys = (c_char_p * special_count)()
      special_values = (c_uint32 * special_count)()
      for i, (token, rank) in enumerate(self._special_tokens.items()):
        special_keys[i] = token.encode('utf-8')
        special_values[i] = rank
    else: special_keys, special_values = None, None

    pattern_buf = create_string_buffer(pattern.encode('utf-8'))
    self.bpe = lib.shredCreate(encoder_keys, encoder_key_lens, encoder_values, len(sorted_items), special_keys, special_values, special_count, pattern_buf)
    if not self.bpe: raise RuntimeError("shredCreate returned NULL")

  def encode(self, text: str, allowed_special: Optional[List[str]] = None) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    if not allowed_special: return self.encode_ordinary(text)
    if allowed_special == "all":  return self._encode_with_special_preprocessing(text, list(self._special_tokens.keys()))
    elif isinstance(allowed_special, str):  return self._encode_with_special_preprocessing(text, [allowed_special])
    else:  return self._encode_with_special_preprocessing(text, allowed_special)

  def _encode_with_special_preprocessing(self, text: str, allowed_special: List[str]) -> List[int]:
    if not allowed_special: return self.encode_ordinary(text)  
    special_pattern = '|'.join(re.escape(token) for token in allowed_special)
    if not special_pattern: return self.encode_ordinary(text)

    parts = re.split(f'({special_pattern})', text)
    tokens = []
    for part in parts:
      if not part: continue
      elif part in self._special_tokens: tokens.append(self._special_tokens[part])
      else: tokens.extend(self.encode_ordinary(part))
    
    return tokens

  def encode_ordinary(self, text: str) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")
    try:
      text_bytes = text.encode('utf-8')
      lib.encodeOrdinary(self.bpe, text_bytes, token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    except:
      return self._fallback_encode(text)
    finally:
      if token_array: lib.tokenArrayFree(token_array)

  def _fallback_encode(self, text: str) -> List[int]:
    tokens = []
    pieces = re.findall(BASIC_REGEX, text)
    for piece in pieces:
      piece_bytes = piece.encode('utf-8')
      if piece_bytes in self._encoder: tokens.append(self._encoder[piece_bytes])
      else:
        for byte in piece_bytes:
          byte_token = bytes([byte])
          if byte_token in self._encoder: tokens.append(self._encoder[byte_token])
          else: tokens.append(0)
    return tokens

  def decode(self, tokens: List[int]) -> str:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    if not tokens: return ""

    byte_array = create_byte_array(lib)
    if not byte_array: raise RuntimeError("Failed to create byte array")

    try:
      tokens_array = (c_uint32 * len(tokens))(*tokens)
      lib.decodeBytes(self.bpe, tokens_array, len(tokens), byte_array)
      if byte_array.contents.len == 0: return ""
      result_bytes = bytes([byte_array.contents.bytes[i] for i in range(byte_array.contents.len)])
      return result_bytes.decode('utf-8', errors='replace')
    finally:
      if byte_array: lib.byteArrayFree(byte_array)

  def encode_unstable(self, text: str, allowed_special: Optional[List[str]] = None):
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    result = create_encode_unstable_result(lib)
    try:
      if allowed_special:
        special_array = (c_char_p * len(allowed_special))(*[s.encode('utf-8') for s in allowed_special])
        lib.encodeWithUnstable(self.bpe, text.encode('utf-8'), special_array, len(allowed_special), result)
      else: lib.encodeWithUnstable(self.bpe, text.encode('utf-8'), None, 0, result)

      tokens = [result.contents.tokens.tokens[i] for i in range(result.contents.tokens.count)]
      completions = []
      for i in range(result.contents.completions.count):
        comp = result.contents.completions.completions[i]
        completions.append([comp.contents.tokens[j] for j in range(comp.contents.count)])
      return {'tokens': tokens, 'completions': completions}
    finally:
      lib.encodeUnstableResultFree(result)

  def encode_bytes(self, data: bytes) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")

    try:
      data_ptr = (c_uint8 * len(data))(*data)
      lib.encodeBytes(self.bpe, data_ptr, len(data), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    finally:
      if token_array: lib.tokenArrayFree(token_array)

  def encode_single_token(self, piece: bytes) -> Optional[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    try:
      piece_ptr = (c_uint8 * len(piece))(*piece)
      result = c_uint32()
      lib.encodeSingleToken(self.bpe, piece_ptr, len(piece), byref(result))
      return result.value
    except:
      return None

  def encode_single_piece(self, piece: bytes) -> List[int]:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    token_array = create_token_array(lib)
    if not token_array: raise RuntimeError("Failed to create token array")

    try:
      piece_ptr = (c_uint8 * len(piece))(*piece)
      lib.encodeSinglePiece(self.bpe, piece_ptr, len(piece), token_array)
      return [token_array.contents.tokens[i] for i in range(token_array.contents.count)]
    finally:
      if token_array: lib.tokenArrayFree(token_array)

  def decode_single_token(self, token: int) -> bytes:
    if not self.bpe: raise RuntimeError("Tokenizer not initialized")
    byte_array = create_byte_array(lib)
    if not byte_array: raise RuntimeError("Failed to create byte array")

    try:
      lib.decodeSingleTokenBytes(self.bpe, token, byte_array)
      if byte_array.contents.len == 0: return b""
      return bytes([byte_array.contents.bytes[i] for i in range(byte_array.contents.len)])
    finally:
      if byte_array: lib.byteArrayFree(byte_array)

  @property
  def vocab_size(self) -> int: 
    if self.bpe: return lib.getTokenCount(self.bpe)
    return len(self._vocab)

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

def load_encoding(encoding_name: str, download: bool = True) -> Shred:
  tokenizer = Shred()
  tokenizer.load_from_encoding(encoding_name, download)
  return tokenizer
import pytest
from shred import Shred, load_encoding

@pytest.fixture
def tokenizer():
  tok = Shred()
  tok.load_from_encoding("base_50k")  # assumes test.model exists remotely
  return tok

def test_load_encoding():
  tok = load_encoding("base_50k")
  assert isinstance(tok, Shred)
  assert tok.vocab_size > 0

def test_encode_decode_roundtrip(tokenizer):
  text = "hello world"
  tokens = tokenizer.encode(text)
  assert isinstance(tokens, list)
  decoded = tokenizer.decode(tokens)
  assert isinstance(decoded, str)
  assert decoded != ""

def test_encode_with_special(tokenizer):
  special = list(tokenizer.special_tokens.keys())
  text = " ".join(special) + " test"
  tokens = tokenizer.encode(text, allowed_special="all")
  assert any(t in tokenizer.special_tokens.values() for t in tokens)

def test_encode_ordinary(tokenizer):
  text = "abcdef12345"
  tokens = tokenizer.encode_ordinary(text)
  assert all(isinstance(t, int) for t in tokens)
  assert len(tokens) > 0

def test_fallback_encode(tokenizer):
  text = "zzzz"
  tokens = tokenizer._fallback_encode(text)
  assert all(isinstance(t, int) for t in tokens)

def test_encode_bytes(tokenizer):
  data = b"byte-data"
  tokens = tokenizer.encode_bytes(data)
  assert len(tokens) > 0

def test_single_token_and_piece(tokenizer):
  piece = b"hello"
  token = tokenizer.encode_single_token(piece)
  assert isinstance(token, int)
  piece_tokens = tokenizer.encode_single_piece(piece)
  assert isinstance(piece_tokens, list)

def test_decode_single_token(tokenizer):
  piece = b"a"
  token = tokenizer.encode_single_token(piece)
  if token is not None:
    decoded = tokenizer.decode_single_token(token)
    assert isinstance(decoded, (bytes, bytearray))

def test_encode_unstable(tokenizer):
  text = "unstable encoding test"
  result = tokenizer.encode_unstable(text)
  assert "tokens" in result
  assert "completions" in result

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
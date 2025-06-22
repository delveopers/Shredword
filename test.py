from shred import Shred, load_encoding

def test_basic_encoding():
  """Test basic encoding/decoding functionality"""
  print("=== Basic Encoding Test ===")

  # Load tokenizer from encoding name
  tokenizer = load_encoding("pre_25k")
  test_text = "Hello world! This is a test of the ShredBPE tokenizer."
  print(f"Original text: {test_text}")
  
  # Encode
  tokens = tokenizer.encode(test_text)
  print(f"Encoded tokens: {tokens}")
  print(f"Token count: {len(tokens)}")
  # Decode back
  decoded = tokenizer.decode(tokens)
  print(f"Decoded text: {decoded}")
  print(f"Round-trip success: {test_text == decoded}")
  # Test vocab size
  print(f"Vocab size: {tokenizer.vocab_size}")

def test_special_tokens():
  """Test encoding with special tokens"""
  print("\n=== Special Tokens Test ===")
  
  tokenizer = load_encoding("pre_16k")
  # Print available special tokens
  print(f"Available special tokens: {tokenizer.special_tokens}")
  # Test with potential special tokens
  test_text = "<s> Hello world! </s>"
  # Use "all" to allow all special tokens, or pass a list of specific ones
  tokens_all = tokenizer.encode(test_text, allowed_special="all")
  tokens_none = tokenizer.encode_ordinary(test_text)  # No special tokens
  print(f"Text: {test_text}")
  print(f"With special tokens: {tokens_all}")
  print(f"Without special tokens: {tokens_none}")

  # Test with specific special tokens if they exist
  if tokenizer.special_tokens:
    specific_tokens = list(tokenizer.special_tokens.keys())[:2]  # First 2 special tokens
    tokens_specific = tokenizer.encode(test_text, allowed_special=specific_tokens)
    print(f"With specific tokens {specific_tokens}: {tokens_specific}")

if __name__ == "__main__":
  try:
    test_basic_encoding()
    test_special_tokens()
    print("\n All tests completed!")
    
  except Exception as e:
    print(f" Test failed: {e}")
    import traceback
    traceback.print_exc()
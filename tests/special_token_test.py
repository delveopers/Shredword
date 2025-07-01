from shred import Shred, load_encoding

def test_fixed_special_tokens():
  """Test the fixed special token functionality"""
  print("=== Testing Fixed Special Tokens ===")
  tokenizer = load_encoding("pre_16k")
  test_cases = [
    "<s> Hello world! </s>",
    "<s>Hello</s>",
    "Hello <s> World </s> End",
    "<pad>Some text<unk>",
    "Regular text without special tokens",
  ]
  
  for test_text in test_cases:
    print(f"\nText: '{test_text}'")

    # Test with special tokens allowed
    tokens_special = tokenizer.encode(test_text, allowed_special="all")
    print(f"  With special tokens: {tokens_special}")

    # Test without special tokens
    tokens_ordinary = tokenizer.encode_ordinary(test_text)
    print(f"  Ordinary encoding:   {tokens_ordinary}")

    # Check if they're different (they should be for texts with special tokens)
    different = tokens_special != tokens_ordinary
    print(f"  Different results: {different}")

    # Decode back to verify
    decoded_special = tokenizer.decode(tokens_special)
    decoded_ordinary = tokenizer.decode(tokens_ordinary)
    print(f"  Decoded (special):   '{decoded_special}'")
    print(f"  Decoded (ordinary):  '{decoded_ordinary}'")

    # Round-trip test
    round_trip_ok = test_text == decoded_special
    print(f"  Round-trip success:  {round_trip_ok}")

def test_individual_special_tokens():
  """Test individual special token encoding"""
  print("\n\n=== Individual Special Token Tests ===")    
  tokenizer = load_encoding("pre_16k")    
  special_tokens_to_test = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]    
  for token in special_tokens_to_test:
    print(f"\nTesting: '{token}'")        
    # Should encode to single token ID when special tokens allowed
    with_special = tokenizer.encode(token, allowed_special="all")
    without_special = tokenizer.encode_ordinary(token)

    print(f"  With special:    {with_special}")
    print(f"  Without special: {without_special}")

    # Verify the special token maps to correct ID
    expected_id = tokenizer.special_tokens.get(token)
    if expected_id is not None:
      success = len(with_special) == 1 and with_special[0] == expected_id
      print(f"  Expected ID: {expected_id}, Got: {with_special}, Success: {success}")
    else:
      print(f"  Token not in special_tokens dict")

if __name__ == "__main__":
  test_fixed_special_tokens()
  test_individual_special_tokens()
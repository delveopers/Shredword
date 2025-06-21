/**
  @file core.h
  @brief Main CoreBPE tokenizer implementation for Byte Pair Encoding

  * This file provides the primary API for BPE (Byte Pair Encoding) tokenization including:
  * - CoreBPE structure containing encoder/decoder mappings and regex patterns
  * - Text encoding functions (ordinary text, special tokens, raw bytes)
  * - Token decoding functions (tokens to bytes, single token decoding)
  * - Utility functions for token management and introspection

  * The CoreBPE tokenizer supports both regular tokens and special tokens with
  * configurable regex patterns for text preprocessing.

  * Compilation command:
  * linux: g++ -std=c++11 -shared -fPIC -o libtoken.so core.cpp token.cpp hash.cpp -lregex
  * windows: g++ -std=c++11 -shared -o libtoken.dll core.cpp token.cpp hash.cpp -lregex

  * Alternative (if regex library unavailable):
  * g++ -std=c++11 -shared -fPIC -o libtoken.so core.cpp token.cpp hash.cpp
*/

#ifndef __SHRED_H__
#define __SHRED_H__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "hash.h"
#include "token.h"

// Main CoreBPE structure
typedef struct {
  HashMap* encoder;
  HashMapStr* special_tokens_encoder;
  ReverseMap* decoder;
  ReverseMap* special_tokens_decoder;
  void* regex;    // regex pointers
  void* special_regex;    // regex pointers
  SortedTokens* sorted_token_bytes;
} CoreBPE;

extern "C" {
  // Core API functions
  CoreBPE* shred_new(const uint8_t** encoder_keys, const size_t* encoder_key_lens, const Rank* encoder_values, size_t encoder_count, const char** special_token_keys, const Rank* special_token_values, size_t special_token_count, const char* pattern);
  void shred_free(CoreBPE* bpe);

  // Encoding functions
  ShredError encode_ordinary(CoreBPE* bpe, const char* text, TokenArray* result);
  ShredError encode(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, TokenArray* result);
  ShredError encode_with_unstable(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, EncodeUnstableResult* result);
  ShredError encode_bytes(CoreBPE* bpe, const uint8_t* bytes, size_t byte_len, TokenArray* result);
  ShredError encode_single_token(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, Rank* result);
  ShredError encode_single_piece(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, TokenArray* result);

  // Decoding functions
  ShredError decode_bytes(CoreBPE* bpe, const Rank* tokens, size_t token_count, ByteArray* result);
  ShredError decode_single_token_bytes(CoreBPE* bpe, Rank token, ByteArray* result);

  // Utility functions
  size_t get_token_count(CoreBPE* bpe);
  ShredError get_token_byte_values(CoreBPE* bpe, ByteArray** results, size_t* count);
  ShredError token_array_push(TokenArray* array, Rank token);
}
#endif //!__SHRED_H__
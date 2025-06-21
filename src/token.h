/**
  @file token.h
  @brief Token array and completion handling structures for BPE tokenization

  * This file provides data structures for:
  * - Token arrays for storing sequences of encoded tokens
  * - Byte arrays for handling raw byte data
  * - Completion sets for managing multiple token completion possibilities
  * - Sorted token structures for efficient prefix-based token searches
  * - Encode unstable results for handling partial/incomplete tokenization
  * Used by the CoreBPE tokenizer for managing token sequences and completions.
*/

#ifndef __TOKEN__H__
#define __TOKEN__H__

#include <stdint.h>
#include <stddef.h>
#include "hash.h"

// Sorted token bytes for completion search
typedef struct {
  uint8_t** tokens;
  size_t* token_lens;
  size_t count;
  size_t capacity;
} SortedTokens;

// Result structures
typedef struct {
  Rank* tokens;
  size_t count;
  size_t capacity;
} TokenArray;

typedef struct {
  TokenArray** completions;
  size_t count;
  size_t capacity;
} CompletionSet;

typedef struct {
  TokenArray tokens;
  CompletionSet completions;
} EncodeUnstableResult;

typedef struct {
  uint8_t* bytes;
  size_t len;
} ByteArray;

extern "C" {
  // Memory management helpers
  TokenArray* token_array_new(size_t capacity);
  void token_array_free(TokenArray* array);
  void token_array_clear(TokenArray* array);

  // completion set functions
  CompletionSet* completion_set_new(size_t capacity);
  void completion_set_free(CompletionSet* set);

  EncodeUnstableResult* encode_unstable_result_new(void);
  void encode_unstable_result_free(EncodeUnstableResult* result);

  ByteArray* byte_array_new(size_t capacity);
  void byte_array_free(ByteArray* array);
  void byte_array_clear(ByteArray* array);

  SortedTokens* sorted_tokens_new(void);
  void sorted_tokens_free(SortedTokens* tokens);
  ShredError sorted_tokens_add(SortedTokens* tokens, const uint8_t* token, size_t token_len);
  ShredError sorted_tokens_sort(SortedTokens* tokens);
  size_t sorted_tokens_find_prefix(SortedTokens* tokens, const uint8_t* prefix, size_t prefix_len);
  ShredError completion_set_add(CompletionSet* set, TokenArray* completion);
}

#endif  //!__TOKEN__H__
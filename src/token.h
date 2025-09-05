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
  size_t *token_lens, count, capacity;
} SortedTokens;

typedef struct {
  Rank* tokens;
  size_t count, capacity;
} TokenArray;

typedef struct {
  TokenArray** completions;
  size_t count, capacity;
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
  TokenArray* tokenArrayCreate(size_t capacity);
  void tokenArrayFree(TokenArray* array);
  void tokenArrayClear(TokenArray* array);

  CompletionSet* completionSetCreate(size_t capacity);
  void completionSetFree(CompletionSet* set);
  void completionSetAdd(CompletionSet* set, TokenArray* completion);

  EncodeUnstableResult* encodeUnstableResultCreate();
  void encodeUnstableResultFree(EncodeUnstableResult* result);

  ByteArray* byteArrayCreate(size_t capacity);
  void byteArrayFree(ByteArray* array);
  void byteArrayClear(ByteArray* array);

  SortedTokens* sortedTokensCreate();
  void sortedTokensFree(SortedTokens* tokens);
  void sortedTokensAdd(SortedTokens* tokens, const uint8_t* token, size_t token_len);
  void sortedTokensSort(SortedTokens* tokens);
  size_t sortedTokensFindPrefix(SortedTokens* tokens, const uint8_t* prefix, size_t prefix_len);
}

#endif  //!__TOKEN__H__
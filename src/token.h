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
}

#endif  //!__TOKEN__H__
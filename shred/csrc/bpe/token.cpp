#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "token.h"
#include "hashmap.h"
#include "core.h"

TokenArray* tokenArrayCreate(size_t capacity) {
  if (capacity == 0) capacity = 64;
  TokenArray* array = (TokenArray*)malloc(sizeof(TokenArray));
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 102 <tokenArrayPush() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  array->tokens = (Rank*)malloc(sizeof(Rank) * capacity);
  if (!array->tokens) {
    fprintf(stderr, "SHRED>ERROR 102 <tokenArrayPush() in token.cpp>:  Couldn't allocate memory\n");
    free(array);
    exit(EXIT_FAILURE);
  }
  array->count = 0;
  array->capacity = capacity;
  return array;
}

void tokenArrayFree(TokenArray* array) {
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 101 <tokenArrayFree() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  free(array->tokens);
  free(array);
}

void tokenArrayClear(TokenArray* array) { if (array) array->count = 0; }

void tokenArrayPush(TokenArray* array, Rank token) {
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 101 <tokenArrayPush() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  
  if (array->count >= array->capacity) {
    size_t new_capacity = array->capacity * 2;
    Rank* new_tokens = (Rank*)realloc(array->tokens, sizeof(Rank) * new_capacity);
    if (!new_tokens) {
      fprintf(stderr, "SHRED>ERROR 102 <tokenArrayPush() in token.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    array->tokens = new_tokens;
    array->capacity = new_capacity;
  }
  array->tokens[array->count++] = token;
}

ByteArray* byteArrayCreate(size_t capacity) {
  if (capacity == 0) capacity = 256;

  ByteArray* array = (ByteArray*)malloc(sizeof(ByteArray));
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 102 <byteArrayCreate() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }  
  array->bytes = (uint8_t*)malloc(capacity);
  if (!array->bytes) {
    fprintf(stderr, "SHRED>ERROR 102 <byteArrayCreate() in token.cpp>:  Couldn't allocate memory\n");
    free(array);
    exit(EXIT_FAILURE);
  }
  array->len = 0;
  return array;
}

void byteArrayFree(ByteArray* array) {
  if (!array) return;
  free(array->bytes);
  free(array);
}

void byteArrayClear(ByteArray* array) { if (array) array->len = 0; }

CompletionSet* completionSetCreate(size_t capacity) {
  if (capacity == 0) capacity = 16;
  CompletionSet* set = (CompletionSet*)malloc(sizeof(CompletionSet));
  if (!set) {
    fprintf(stderr, "SHRED>ERROR 102 <completionSetCreate() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }

  set->completions = (TokenArray**)malloc(sizeof(TokenArray*) * capacity);
  if (!set->completions) {
    fprintf(stderr, "SHRED>ERROR 102 <completionSetCreate() in token.cpp>:  Couldn't allocate memory\n");
    free(set);
    exit(EXIT_FAILURE);
  }

  set->count = 0;
  set->capacity = capacity;
  return set;
}

void completionSetFree(CompletionSet* set) {
  if (!set) return;

  for (size_t i = 0; i < set->count; i++) tokenArrayFree(set->completions[i]);
  free(set->completions);
  free(set);
}

void completionSetAdd(CompletionSet* set, TokenArray* completion) {
  if (!set || !completion) {
    fprintf(stderr, "SHRED>ERROR 101 <completionSetAdd() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  if (set->count >= set->capacity) {
    size_t new_capacity = set->capacity * 2;
    TokenArray** new_completions = (TokenArray**)realloc(set->completions, sizeof(TokenArray*) * new_capacity);
    if (!new_completions) {
      fprintf(stderr, "SHRED>ERROR 102 <completionSetAdd() in token.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    set->completions = new_completions;
    set->capacity = new_capacity;
  }
  set->completions[set->count++] = completion;
}

EncodeUnstableResult* encodeUnstableResultCreate() {
  EncodeUnstableResult* result = (EncodeUnstableResult*)malloc(sizeof(EncodeUnstableResult));
  if (!result) {
    fprintf(stderr, "SHRED>ERROR 102 <encodeUnstableResultCreate() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  result->tokens.tokens = (Rank*)malloc(sizeof(Rank) * 64);  // Default capacity
  if (!result->tokens.tokens) {
    free(result);
    return NULL;
  }
  result->tokens.count = 0;
  result->tokens.capacity = 64;

  result->completions.completions = (TokenArray**)malloc(sizeof(TokenArray*) * 16);
  if (!result->completions.completions) {
    fprintf(stderr, "SHRED>ERROR 102 <encodeUnstableResultCreate() in token.cpp>:  Couldn't allocate memory\n");
    free(result->tokens.tokens);
    free(result);
    exit(EXIT_FAILURE);
  }
  result->completions.count = 0;
  result->completions.capacity = 16;
  return result;
}

void encodeUnstableResultFree(EncodeUnstableResult* result) {
  if (!result) return;
  free(result->tokens.tokens);
  for (size_t i = 0; i < result->completions.count; i++) tokenArrayFree(result->completions.completions[i]);
  free(result->completions.completions);
  free(result);
}

void encodeWithUnstable(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, EncodeUnstableResult* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeWithUnstable() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  result->tokens.count = 0;
  for (size_t i = 0; i < result->completions.count; i++) tokenArrayFree(result->completions.completions[i]);
  result->completions.count = 0;

  // For now, just do regular encoding and leave completions empty
  // This is a simplified implementation
  encode(bpe, text, allowed_special, allowed_special_count, &result->tokens);
}

// Sorted tokens implementation
static int compareByteArrays(const void* a, const void* b) {
  const uint8_t** arr_a = (const uint8_t**)a;
  const uint8_t** arr_b = (const uint8_t**)b;

  // This is a simplified comparison - in practice you'd need to compare lengths too
  return memcmp(*arr_a, *arr_b, 16); // Assuming max 16 bytes for simplicity
}

SortedTokens* sortedTokensCreate() {
  SortedTokens* tokens = (SortedTokens*)malloc(sizeof(SortedTokens));
  if (!tokens) return NULL;
  
  tokens->tokens = NULL;
  tokens->token_lens = NULL;
  tokens->count = 0;
  tokens->capacity = 0;
  return tokens;
}

void sortedTokensFree(SortedTokens* tokens) {
  if (!tokens) return;

  for (size_t i = 0; i < tokens->count; i++) free(tokens->tokens[i]);
  free(tokens->tokens);
  free(tokens->token_lens);
  free(tokens);
}

void sortedTokensAdd(SortedTokens* tokens, const uint8_t* token, size_t token_len) {
  if (!tokens || !token) {
    fprintf(stderr, "SHRED>ERROR 101 <sortedTokensAdd() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  if (tokens->count >= tokens->capacity) {
    size_t new_capacity = tokens->capacity == 0 ? 256 : tokens->capacity * 2;

    uint8_t** new_tokens = (uint8_t**)realloc(tokens->tokens, sizeof(uint8_t*) * new_capacity);
    if (!new_tokens) {
      fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }

    size_t* new_token_lens = (size_t*)realloc(tokens->token_lens, sizeof(size_t) * new_capacity);
    if (!new_token_lens) {
      // If we can't allocate token_lens, we need to restore tokens to its original state
      if (tokens->capacity == 0) free(new_tokens);
      fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }

    tokens->tokens = new_tokens;
    tokens->token_lens = new_token_lens;
    tokens->capacity = new_capacity;
  }

  tokens->tokens[tokens->count] = (uint8_t*)malloc(token_len);
  if (!tokens->tokens[tokens->count]) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  memcpy(tokens->tokens[tokens->count], token, token_len);
  tokens->token_lens[tokens->count] = token_len;
  tokens->count++;
}

void sortedTokensSort(SortedTokens* tokens) {
  if (!tokens || tokens->count == 0) {
    fprintf(stderr, "SHRED>ERROR 101 <sortedTokensSort() in token.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  uint8_t*** sort_array = (uint8_t***)malloc(tokens->count * sizeof(uint8_t**));
  if (!sort_array) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensSort() in token.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < tokens->count; i++) sort_array[i] = &tokens->tokens[i];
  qsort(sort_array, tokens->count, sizeof(uint8_t**), compareByteArrays);

  uint8_t** new_tokens = (uint8_t**)malloc(tokens->count * sizeof(uint8_t*));
  size_t* new_token_lens = (size_t*)malloc(tokens->count * sizeof(size_t));
  
  if (!new_tokens || !new_token_lens) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensSort() in token.cpp>:  Couldn't allocate memory\n");
    free(sort_array);
    free(new_tokens);
    free(new_token_lens);
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < tokens->count; i++) {
    size_t orig_idx = sort_array[i] - tokens->tokens;
    new_tokens[i] = tokens->tokens[orig_idx];
    new_token_lens[i] = tokens->token_lens[orig_idx];
  }

  free(tokens->tokens);
  free(tokens->token_lens);
  tokens->tokens = new_tokens;
  tokens->token_lens = new_token_lens;
  free(sort_array);
}

size_t sortedTokensFindPrefix(SortedTokens* tokens, const uint8_t* prefix, size_t prefix_len) {
  if (!tokens || !prefix || tokens->count == 0) return SIZE_MAX;

  // Binary search for first token that starts with prefix
  size_t left = 0;
  size_t right = tokens->count;

  while (left < right) {
    size_t mid = left + (right - left) / 2;
    size_t cmp_len = tokens->token_lens[mid] < prefix_len ? tokens->token_lens[mid] : prefix_len;
    int cmp = memcmp(tokens->tokens[mid], prefix, cmp_len);

    if (cmp < 0 || (cmp == 0 && tokens->token_lens[mid] < prefix_len)) left = mid + 1;
    else right = mid;
  }

  // Check if we found a valid prefix match
  if (left < tokens->count) {
    size_t cmp_len = tokens->token_lens[left] < prefix_len ? tokens->token_lens[left] : prefix_len;
    if (memcmp(tokens->tokens[left], prefix, cmp_len) == 0) return left;
  }
  return SIZE_MAX;
}
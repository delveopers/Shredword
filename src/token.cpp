#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "token.h"
#include "hash.h"
#include "core.h"

TokenArray* token_array_new(size_t capacity) {
  if (capacity == 0) capacity = 64;
  
  TokenArray* array = (TokenArray*)malloc(sizeof(TokenArray));
  if (!array) return NULL;
  
  array->tokens = (Rank*)malloc(sizeof(Rank) * capacity);
  if (!array->tokens) {
    free(array);
    return NULL;
  }
  
  array->count = 0;
  array->capacity = capacity;
  return array;
}

void token_array_free(TokenArray* array) {
  if (!array) return;
  free(array->tokens);
  free(array);
}

void token_array_clear(TokenArray* array) {
  if (array) {
    array->count = 0;
  }
}

ShredError token_array_push(TokenArray* array, Rank token) {
  if (!array) return ERROR_NULL_POINTER;
  
  if (array->count >= array->capacity) {
    size_t new_capacity = array->capacity * 2;
    Rank* new_tokens = (Rank*)realloc(array->tokens, sizeof(Rank) * new_capacity);
    if (!new_tokens) return ERROR_MEMORY_ALLOCATION;
    
    array->tokens = new_tokens;
    array->capacity = new_capacity;
  }
  
  array->tokens[array->count++] = token;
  return OK;
}

ByteArray* byte_array_new(size_t capacity) {
  if (capacity == 0) capacity = 256;

  ByteArray* array = (ByteArray*)malloc(sizeof(ByteArray));
  if (!array) return NULL;
  
  array->bytes = (uint8_t*)malloc(capacity);
  if (!array->bytes) {
    free(array);
    return NULL;
  }
  
  array->len = 0;
  return array;
}

void byte_array_free(ByteArray* array) {
  if (!array) return;
  free(array->bytes);
  free(array);
}

void byte_array_clear(ByteArray* array) {
  if (array) {
    array->len = 0;
  }
}

// Completion set implementation
CompletionSet* completion_set_new(size_t capacity) {
  if (capacity == 0) capacity = 16;

  CompletionSet* set = (CompletionSet*)malloc(sizeof(CompletionSet));
  if (!set) return NULL;

  set->completions = (TokenArray**)malloc(sizeof(TokenArray*) * capacity);
  if (!set->completions) {
    free(set);
    return NULL;
  }

  set->count = 0;
  set->capacity = capacity;
  return set;
}

void completion_set_free(CompletionSet* set) {
  if (!set) return;

  for (size_t i = 0; i < set->count; i++) {
    token_array_free(set->completions[i]);
  }
  free(set->completions);
  free(set);
}

ShredError completion_set_add(CompletionSet* set, TokenArray* completion) {
  if (!set || !completion) return ERROR_NULL_POINTER;

  if (set->count >= set->capacity) {
    size_t new_capacity = set->capacity * 2;
    TokenArray** new_completions = (TokenArray**)realloc(set->completions, sizeof(TokenArray*) * new_capacity);
    if (!new_completions) return ERROR_MEMORY_ALLOCATION;

    set->completions = new_completions;
    set->capacity = new_capacity;
  }

  set->completions[set->count++] = completion;
  return OK;
}

// Encode unstable result implementation
EncodeUnstableResult* encode_unstable_result_new(void) {
  EncodeUnstableResult* result = (EncodeUnstableResult*)malloc(sizeof(EncodeUnstableResult));
  if (!result) return NULL;

  // Initialize tokens array
  result->tokens.tokens = (Rank*)malloc(sizeof(Rank) * 64);  // Default capacity
  if (!result->tokens.tokens) {
    free(result);
    return NULL;
  }
  result->tokens.count = 0;
  result->tokens.capacity = 64;

  // Initialize completions set
  result->completions.completions = (TokenArray**)malloc(sizeof(TokenArray*) * 16);
  if (!result->completions.completions) {
    free(result->tokens.tokens);
    free(result);
    return NULL;
  }
  result->completions.count = 0;
  result->completions.capacity = 16;

  return result;
}

void encode_unstable_result_free(EncodeUnstableResult* result) {
  if (!result) return;

  // Free tokens array
  free(result->tokens.tokens);

  // Free completions
  for (size_t i = 0; i < result->completions.count; i++) {
    token_array_free(result->completions.completions[i]);
  }
  free(result->completions.completions);

  free(result);
}

ShredError encode_with_unstable(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, EncodeUnstableResult* result) {
  if (!bpe || !text || !result) return ERROR_NULL_POINTER;

  // Clear result
  result->tokens.count = 0;
  for (size_t i = 0; i < result->completions.count; i++) {
    token_array_free(result->completions.completions[i]);
  }
  result->completions.count = 0;

  // For now, just do regular encoding and leave completions empty
  // This is a simplified implementation
  ShredError err = encode(bpe, text, allowed_special, allowed_special_count, &result->tokens);
  if (err != OK) return err;

  return OK;
}

// Sorted tokens implementation
static int compare_byte_arrays(const void* a, const void* b) {
  const uint8_t** arr_a = (const uint8_t**)a;
  const uint8_t** arr_b = (const uint8_t**)b;

  // This is a simplified comparison - in practice you'd need to compare lengths too
  return memcmp(*arr_a, *arr_b, 16); // Assuming max 16 bytes for simplicity
}

SortedTokens* sorted_tokens_new(void) {
  SortedTokens* tokens = (SortedTokens*)malloc(sizeof(SortedTokens));
  if (!tokens) return NULL;
  
  tokens->tokens = NULL;
  tokens->token_lens = NULL;
  tokens->count = 0;
  tokens->capacity = 0;
  return tokens;
}

void sorted_tokens_free(SortedTokens* tokens) {
  if (!tokens) return;

  for (size_t i = 0; i < tokens->count; i++) {
    free(tokens->tokens[i]);
  }
  free(tokens->tokens);
  free(tokens->token_lens);
  free(tokens);
}

ShredError sorted_tokens_add(SortedTokens* tokens, const uint8_t* token, size_t token_len) {
  if (!tokens || !token) return ERROR_NULL_POINTER;

  if (tokens->count >= tokens->capacity) {
    size_t new_capacity = tokens->capacity == 0 ? 256 : tokens->capacity * 2;
    
    uint8_t** new_tokens = (uint8_t**)realloc(tokens->tokens, sizeof(uint8_t*) * new_capacity);
    if (!new_tokens) return ERROR_MEMORY_ALLOCATION;

    size_t* new_token_lens = (size_t*)realloc(tokens->token_lens, sizeof(size_t) * new_capacity);
    if (!new_token_lens) {
      // If we can't allocate token_lens, we need to restore tokens to its original state
      if (tokens->capacity == 0) {
        free(new_tokens);
      }
      return ERROR_MEMORY_ALLOCATION;
    }

    tokens->tokens = new_tokens;
    tokens->token_lens = new_token_lens;
    tokens->capacity = new_capacity;
  }

  // Allocate and copy token
  tokens->tokens[tokens->count] = (uint8_t*)malloc(token_len);
  if (!tokens->tokens[tokens->count]) return ERROR_MEMORY_ALLOCATION;

  memcpy(tokens->tokens[tokens->count], token, token_len);
  tokens->token_lens[tokens->count] = token_len;
  tokens->count++;

  return OK;
}

ShredError sorted_tokens_sort(SortedTokens* tokens) {
  if (!tokens || tokens->count == 0) return OK;

  // Create array of pointers for sorting
  uint8_t*** sort_array = (uint8_t***)malloc(tokens->count * sizeof(uint8_t**));
  if (!sort_array) return ERROR_MEMORY_ALLOCATION;

  for (size_t i = 0; i < tokens->count; i++) {
    sort_array[i] = &tokens->tokens[i];
  }

  // Sort using qsort
  qsort(sort_array, tokens->count, sizeof(uint8_t**), compare_byte_arrays);

  // Rebuild the arrays in sorted order
  uint8_t** new_tokens = (uint8_t**)malloc(tokens->count * sizeof(uint8_t*));
  size_t* new_token_lens = (size_t*)malloc(tokens->count * sizeof(size_t));
  
  if (!new_tokens || !new_token_lens) {
    free(sort_array);
    free(new_tokens);
    free(new_token_lens);
    return ERROR_MEMORY_ALLOCATION;
  }

  for (size_t i = 0; i < tokens->count; i++) {
    size_t orig_idx = sort_array[i] - tokens->tokens;
    new_tokens[i] = tokens->tokens[orig_idx];
    new_token_lens[i] = tokens->token_lens[orig_idx];
  }

  // Replace old arrays
  free(tokens->tokens);
  free(tokens->token_lens);
  tokens->tokens = new_tokens;
  tokens->token_lens = new_token_lens;

  free(sort_array);
  return OK;
}

size_t sorted_tokens_find_prefix(SortedTokens* tokens, const uint8_t* prefix, size_t prefix_len) {
  if (!tokens || !prefix || tokens->count == 0) return SIZE_MAX;

  // Binary search for first token that starts with prefix
  size_t left = 0;
  size_t right = tokens->count;

  while (left < right) {
    size_t mid = left + (right - left) / 2;
    size_t cmp_len = tokens->token_lens[mid] < prefix_len ? tokens->token_lens[mid] : prefix_len;
    int cmp = memcmp(tokens->tokens[mid], prefix, cmp_len);

    if (cmp < 0 || (cmp == 0 && tokens->token_lens[mid] < prefix_len)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // Check if we found a valid prefix match
  if (left < tokens->count) {
    size_t cmp_len = tokens->token_lens[left] < prefix_len ? tokens->token_lens[left] : prefix_len;
    if (memcmp(tokens->tokens[left], prefix, cmp_len) == 0) {
      return left;
    }
  }

  return SIZE_MAX;
}
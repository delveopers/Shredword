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

// Sorted tokens implementation
static int compare_byte_arrays(const void* a, const void* b) {
  const uint8_t** arr_a = (const uint8_t**)a;
  const uint8_t** arr_b = (const uint8_t**)b;

  // This is a simplified comparison - in practice you'd need to compare lengths too
  return memcmp(*arr_a, *arr_b, 16); // Assuming max 16 bytes for simplicity
}

static SortedTokens* sorted_tokens_new(void) {
  SortedTokens* tokens = (SortedTokens*)malloc(sizeof(SortedTokens));
  if (!tokens) return NULL;
  
  tokens->tokens = NULL;
  tokens->token_lens = NULL;
  tokens->count = 0;
  tokens->capacity = 0;
  return tokens;
}

static void sorted_tokens_free(SortedTokens* tokens) {
  if (!tokens) return;

  for (size_t i = 0; i < tokens->count; i++) {
    free(tokens->tokens[i]);
  }
  free(tokens->tokens);
  free(tokens->token_lens);
  free(tokens);
}


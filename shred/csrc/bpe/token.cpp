#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "token.h"
#include "hashmap.h"
#include "core.h"

// TokenArray (dynamic array of Rank)
TokenArray* tokenArrayCreate(size_t capacity) {
  if (capacity == 0) capacity = 128;
  TokenArray* array = (TokenArray*)malloc(sizeof(TokenArray));
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 102 <tokenArrayCreate() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  array->tokens = (Rank*)malloc(sizeof(Rank) * capacity);
  if (!array->tokens) {
    fprintf(stderr, "SHRED>ERROR 102 <tokenArrayCreate() in token.c>:  Couldn't allocate memory\n");
    free(array);
    exit(EXIT_FAILURE);
  }
  array->count = 0;
  array->capacity = capacity;
  return array;
}

void tokenArrayFree(TokenArray* array) {
  if (!array) return;
  free(array->tokens);
  free(array);
}

void tokenArrayClear(TokenArray* array) { if (array) array->count = 0; }

void tokenArrayPush(TokenArray* array, Rank token) {
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 101 <tokenArrayPush() in token.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (array->count >= array->capacity) {
    size_t new_capacity = array->capacity ? array->capacity * 2 : 128;
    Rank* new_tokens = (Rank*)realloc(array->tokens, sizeof(Rank) * new_capacity);
    if (!new_tokens) {
      fprintf(stderr, "SHRED>ERROR 102 <tokenArrayPush() in token.c>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    array->tokens = new_tokens;
    array->capacity = new_capacity;
  }
  array->tokens[array->count++] = token;
}

// ByteArray (dynamic bytes buffer)
ByteArray* byteArrayCreate(size_t capacity) {
  if (capacity == 0) capacity = 512;
  ByteArray* array = (ByteArray*)malloc(sizeof(ByteArray));
  if (!array) {
    fprintf(stderr, "SHRED>ERROR 102 <byteArrayCreate() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  array->bytes = (uint8_t*)malloc(capacity ? capacity : 1);
  if (!array->bytes) {
    fprintf(stderr, "SHRED>ERROR 102 <byteArrayCreate() in token.c>:  Couldn't allocate memory\n");
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

// CompletionSet (array of TokenArray*)
CompletionSet* completionSetCreate(size_t capacity) {
  if (capacity == 0) capacity = 16;
  CompletionSet* set = (CompletionSet*)malloc(sizeof(CompletionSet));
  if (!set) {
    fprintf(stderr, "SHRED>ERROR 102 <completionSetCreate() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  set->completions = (TokenArray**)malloc(sizeof(TokenArray*) * capacity);
  if (!set->completions) {
    fprintf(stderr, "SHRED>ERROR 102 <completionSetCreate() in token.c>:  Couldn't allocate memory\n");
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
    fprintf(stderr, "SHRED>ERROR 101 <completionSetAdd() in token.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (set->count >= set->capacity) {
    size_t new_capacity = set->capacity ? set->capacity * 2 : 16;
    TokenArray** new_completions = (TokenArray**)realloc(set->completions, sizeof(TokenArray*) * new_capacity);
    if (!new_completions) {
      fprintf(stderr, "SHRED>ERROR 102 <completionSetAdd() in token.c>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    set->completions = new_completions;
    set->capacity = new_capacity;
  }
  set->completions[set->count++] = completion;
}

// EncodeUnstableResult helpers
EncodeUnstableResult* encodeUnstableResultCreate() {
  EncodeUnstableResult* result = (EncodeUnstableResult*)malloc(sizeof(EncodeUnstableResult));
  if (!result) {
    fprintf(stderr, "SHRED>ERROR 102 <encodeUnstableResultCreate() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  result->tokens.tokens = (Rank*)malloc(sizeof(Rank) * 128);
  if (!result->tokens.tokens) {
    free(result);
    fprintf(stderr, "SHRED>ERROR 102 <encodeUnstableResultCreate() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  result->tokens.count = 0;
  result->tokens.capacity = 128;
  result->completions.completions = (TokenArray**)malloc(sizeof(TokenArray*) * 16);
  if (!result->completions.completions) {
    fprintf(stderr, "SHRED>ERROR 102 <encodeUnstableResultCreate() in token.c>:  Couldn't allocate memory\n");
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
    fprintf(stderr, "SHRED>ERROR 101 <encodeWithUnstable() in token.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  // reset tokens/completions for the result - keep allocated memory for reuse
  result->tokens.count = 0;
  for (size_t i = 0; i < result->completions.count; i++) tokenArrayFree(result->completions.completions[i]);
  result->completions.count = 0;
  // call core encode (assumes encode will push tokens into result->tokens using tokenArrayPush)
  encode(bpe, text, allowed_special, allowed_special_count, &result->tokens);
}

// SortedTokens: dynamic list of token byte pointers with lengths, sorted lexicographically
// creating a small struct for sorting comparison to make comparator simple and correct
typedef struct {
  uint8_t* ptr;
  size_t len;
} token_pair_t;

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
    fprintf(stderr, "SHRED>ERROR 101 <sortedTokensAdd() in token.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (tokens->count >= tokens->capacity) {
    size_t new_capacity = tokens->capacity == 0 ? 512 : tokens->capacity * 2;
    uint8_t** new_tokens = (uint8_t**)realloc(tokens->tokens, sizeof(uint8_t*) * new_capacity);
    if (!new_tokens) {
      fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.c>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    size_t* new_token_lens = (size_t*)realloc(tokens->token_lens, sizeof(size_t) * new_capacity);
    if (!new_token_lens) {
      if (tokens->capacity == 0) free(new_tokens);
      fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.c>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    tokens->tokens = new_tokens;
    tokens->token_lens = new_token_lens;
    tokens->capacity = new_capacity;
  }
  tokens->tokens[tokens->count] = (uint8_t*)malloc(token_len ? token_len : 1);
  if (!tokens->tokens[tokens->count]) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensAdd() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  if (token_len) memcpy(tokens->tokens[tokens->count], token, token_len);
  tokens->token_lens[tokens->count] = token_len;
  tokens->count++;
}

static int token_pair_cmp(const void* a, const void* b) {
  const token_pair_t* A = (const token_pair_t*)a;
  const token_pair_t* B = (const token_pair_t*)b;
  size_t min_len = (A->len < B->len) ? A->len : B->len;
  int cmp = memcmp(A->ptr, B->ptr, min_len);
  if (cmp != 0) return cmp;
  if (A->len < B->len) return -1;
  if (A->len > B->len) return 1;
  return 0;
}

void sortedTokensSort(SortedTokens* tokens) {
  if (!tokens) {
    fprintf(stderr, "SHRED>ERROR 101 <sortedTokensSort() in token.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (tokens->count == 0) return;
  token_pair_t* pairs = (token_pair_t*)malloc(tokens->count * sizeof(token_pair_t));
  if (!pairs) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensSort() in token.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < tokens->count; ++i) {
    pairs[i].ptr = tokens->tokens[i];
    pairs[i].len = tokens->token_lens[i];
  }
  qsort(pairs, tokens->count, sizeof(token_pair_t), token_pair_cmp);

  // rewrite tokens/token_lens according to sorted pairs
  uint8_t** new_tokens = (uint8_t**)malloc(tokens->count * sizeof(uint8_t*));
  size_t* new_token_lens = (size_t*)malloc(tokens->count * sizeof(size_t));
  if (!new_tokens || !new_token_lens) {
    fprintf(stderr, "SHRED>ERROR 102 <sortedTokensSort() in token.c>:  Couldn't allocate memory\n");
    free(pairs);
    free(new_tokens);
    free(new_token_lens);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < tokens->count; ++i) {
    new_tokens[i] = pairs[i].ptr;
    new_token_lens[i] = pairs[i].len;
  }
  free(pairs);

  free(tokens->tokens);
  free(tokens->token_lens);
  tokens->tokens = new_tokens;
  tokens->token_lens = new_token_lens;
}

size_t sortedTokensFindPrefix(SortedTokens* tokens, const uint8_t* prefix, size_t prefix_len) {
  if (!tokens || !prefix || tokens->count == 0) return SIZE_MAX;
  size_t left = 0, right = tokens->count;
  while (left < right) {
    size_t mid = left + (right - left) / 2;
    const uint8_t* mid_ptr = tokens->tokens[mid];
    size_t mid_len = tokens->token_lens[mid];
    size_t cmp_len = (mid_len < prefix_len) ? mid_len : prefix_len;
    int cmp = memcmp(mid_ptr, prefix, cmp_len);
    if (cmp < 0 || (cmp == 0 && mid_len < prefix_len)) { left = mid + 1; } else { right = mid; }
  }

  if (left < tokens->count) {
    if (tokens->token_lens[left] >= prefix_len &&
        memcmp(tokens->tokens[left], prefix, prefix_len) == 0) {
      return left;
    }
  }
  return SIZE_MAX;
}
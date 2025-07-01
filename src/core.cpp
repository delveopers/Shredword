#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <regex.h>
#include "hash.h"
#include "token.h"
#include "core.h"

#define C_UINT32_MAX 0xFFFFFFFF

// Forward declarations for internal functions
static ShredError byte_pair_merge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count);
static ShredError byte_pair_encode_internal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result);
static ShredError compile_regex(const char* pattern, regex_t* regex);
static ShredError find_regex_matches(regex_t* regex, const char* text, size_t** matches, size_t* match_count);

// Create new CoreBPE instance
CoreBPE* shred_new(const uint8_t** encoder_keys, const size_t* encoder_key_lens, const Rank* encoder_values, size_t encoder_count, const char** special_token_keys, const Rank* special_token_values, size_t special_token_count, const char* pattern) {
  if (!encoder_keys || !encoder_key_lens || !encoder_values || !pattern) {
    return NULL;
  }

  CoreBPE* bpe = (CoreBPE*)malloc(sizeof(CoreBPE));
  if (!bpe) return NULL;

  // Initialize all fields to NULL/zero first
  memset(bpe, 0, sizeof(CoreBPE));

  // Create encoder hash map
  bpe->encoder = hashmap_new(encoder_count * 2);
  if (!bpe->encoder) {
    shred_free(bpe);
    return NULL;
  }

  // Populate encoder
  for (size_t i = 0; i < encoder_count; i++) {
    if (hashmap_insert(bpe->encoder, encoder_keys[i], encoder_key_lens[i], encoder_values[i]) != OK) {
      shred_free(bpe);
      return NULL;
    }
  }

  // Create decoder reverse map
  bpe->decoder = reverse_map_new(encoder_count * 2);
  if (!bpe->decoder) {
    shred_free(bpe);
    return NULL;
  }

  // Populate decoder
  for (size_t i = 0; i < encoder_count; i++) {
    if (reverse_map_insert(bpe->decoder, encoder_values[i], encoder_keys[i], encoder_key_lens[i]) != OK) {
      shred_free(bpe);
      return NULL;
    }
  }

  // Create special tokens encoder if provided
  if (special_token_keys && special_token_values && special_token_count > 0) {
    bpe->special_tokens_encoder = hashmap_str_new(special_token_count * 2);
    if (!bpe->special_tokens_encoder) {
      shred_free(bpe);
      return NULL;
    }

    bpe->special_tokens_decoder = reverse_map_new(special_token_count * 2);
    if (!bpe->special_tokens_decoder) {
      shred_free(bpe);
      return NULL;
    }

    for (size_t i = 0; i < special_token_count; i++) {
      if (hashmap_str_insert(bpe->special_tokens_encoder, special_token_keys[i], special_token_values[i]) != OK) {
        shred_free(bpe);
        return NULL;
      }

      size_t key_len = strlen(special_token_keys[i]);
      if (reverse_map_insert(bpe->special_tokens_decoder, special_token_values[i], (const uint8_t*)special_token_keys[i], key_len) != OK) {
        shred_free(bpe);
        return NULL;
      }
    }
  }

  // Compile regex
  bpe->regex = malloc(sizeof(regex_t));
  if (!bpe->regex) {
    shred_free(bpe);
    return NULL;
  }

  if (compile_regex(pattern, (regex_t*)bpe->regex) != OK) {
    shred_free(bpe);
    return NULL;
  }

  return bpe;
}

// Free CoreBPE instance
void shred_free(CoreBPE* bpe) {
  if (!bpe) return;

  if (bpe->encoder) {
    hashmap_free(bpe->encoder);
  }
  if (bpe->special_tokens_encoder) {
    hashmap_str_free(bpe->special_tokens_encoder);
  }
  if (bpe->decoder) {
    reverse_map_free(bpe->decoder);
  }
  if (bpe->special_tokens_decoder) {
    reverse_map_free(bpe->special_tokens_decoder);
  }
  if (bpe->regex) {
    regfree((regex_t*)bpe->regex);
    free(bpe->regex);
  }
  if (bpe->special_regex) {
    regfree((regex_t*)bpe->special_regex);
    free(bpe->special_regex);
  }
  if (bpe->sorted_token_bytes) {
    sorted_tokens_free(bpe->sorted_token_bytes);
  }

  free(bpe);
}

// Encode ordinary text (no special tokens)
ShredError encode_ordinary(CoreBPE* bpe, const char* text, TokenArray* result) {
  if (!bpe || !text || !result) return ERROR_NULL_POINTER;

  token_array_clear(result);
  size_t* matches = NULL;
  size_t match_count = 0;
  ShredError err = find_regex_matches((regex_t*)bpe->regex, text, &matches, &match_count);
  if (err != OK) return err;

  for (size_t i = 0; i < match_count; i += 2) {
    size_t start = matches[i];
    size_t end = matches[i + 1];
    size_t piece_len = end - start;
    const uint8_t* piece = (const uint8_t*)(text + start);

    // Try direct lookup first
    Rank token;
    if (hashmap_get(bpe->encoder, piece, piece_len, &token)) {
      err = token_array_push(result, token);
      if (err != OK) {
        free(matches);
        return err;
      }
    } else {
      // Use BPE encoding
      err = byte_pair_encode_internal(piece, piece_len, bpe->encoder, result);
      if (err != OK) {
        free(matches);
        return err;
      }
    }
  }

  free(matches);
  return OK;
}

// Encode with special tokens support
ShredError encode(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, TokenArray* result) {
  if (!bpe || !text || !result) return ERROR_NULL_POINTER;

  // if no special tokens are allowed, use encode_ordinary
  if (!allowed_special || allowed_special_count == 0) { return encode_ordinary(bpe, text, result); }
  // If no special tokens encoder is available, fall back to ordinary encoding
  if (!bpe->special_tokens_encoder) { return encode_ordinary(bpe, text, result); }
  token_array_clear(result);  
  const char* current = text;
  size_t text_len = strlen(text);
  while (current < text + text_len) {
    // Find the next special token occurrence
    const char* next_special = NULL;
    size_t special_len = 0;
    Rank special_token = 0;
    // Check for allowed special tokens at current position
    for (size_t i = 0; i < allowed_special_count; i++) {
      size_t token_len = strlen(allowed_special[i]);
      if (current + token_len <= text + text_len && 
          strncmp(current, allowed_special[i], token_len) == 0) {
        // Found a special token, check if it's the earliest one
        if (!next_special || current < next_special) {
          next_special = current;
          special_len = token_len;
          if (!hashmap_str_get(bpe->special_tokens_encoder, allowed_special[i], &special_token)) {
            // Special token not found in encoder, skip it
            continue;
          }
        }
      }
    }

    if (next_special == current) {
      // Encode the special token
      ShredError err = token_array_push(result, special_token);
      if (err != OK) return err;
      current += special_len;
    } else {
      // Find the next special token in the remaining text
      const char* next_occurrence = NULL;
      size_t next_occurrence_len = 0;
      for (size_t i = 0; i < allowed_special_count; i++) {
        const char* found = strstr(current, allowed_special[i]);
        if (found && (!next_occurrence || found < next_occurrence)) {
          next_occurrence = found;
          next_occurrence_len = strlen(allowed_special[i]);
        }
      }

      // Encode ordinary text up to the next special token (or end of string)
      const char* end_pos = next_occurrence ? next_occurrence : (text + text_len);
      size_t ordinary_len = end_pos - current;
      if (ordinary_len > 0) {
        // Create temporary null-terminated string for ordinary encoding
        char* ordinary_text = (char*)malloc(ordinary_len + 1);
        if (!ordinary_text) return ERROR_MEMORY_ALLOCATION;

        memcpy(ordinary_text, current, ordinary_len);
        ordinary_text[ordinary_len] = '\0';
        ShredError err = encode_ordinary(bpe, ordinary_text, result);
        free(ordinary_text);
        if (err != OK) return err;     
        current = end_pos;
      }
    }
  }
  return OK;
}
// Encode bytes directly
ShredError encode_bytes(CoreBPE* bpe, const uint8_t* bytes, size_t byte_len, TokenArray* result) {
  if (!bpe || !bytes || !result) return ERROR_NULL_POINTER;

  token_array_clear(result);
  return byte_pair_encode_internal(bytes, byte_len, bpe->encoder, result);
}

// Encode single token
ShredError encode_single_token(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, Rank* result) {
  if (!bpe || !piece || !result) return ERROR_NULL_POINTER;
  if (hashmap_get(bpe->encoder, piece, piece_len, result)) { return OK; } // Try regular encoder first
  // Try special tokens encoder if available
  if (bpe->special_tokens_encoder) {
    // Convert to null-terminated string for special token lookup
    char* piece_str = (char*)malloc(piece_len + 1);
    if (!piece_str) return ERROR_MEMORY_ALLOCATION;

    memcpy(piece_str, piece, piece_len);
    piece_str[piece_len] = '\0';
    bool found = hashmap_str_get(bpe->special_tokens_encoder, piece_str, result);
    free(piece_str);
    if (found) return OK;
  }

  return ERROR_INVALID_TOKEN;
}

// Encode single piece with BPE
ShredError encode_single_piece(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, TokenArray* result) {
  if (!bpe || !piece || !result) return ERROR_NULL_POINTER;

  token_array_clear(result);
  Rank token;   // Try direct lookup first
  if (hashmap_get(bpe->encoder, piece, piece_len, &token)) { return token_array_push(result, token); }
  return byte_pair_encode_internal(piece, piece_len, bpe->encoder, result); // Use BPE encoding
}

// Decode tokens to bytes
ShredError decode_bytes(CoreBPE* bpe, const Rank* tokens, size_t token_count, ByteArray* result) {
  if (!bpe || !tokens || !result) return ERROR_NULL_POINTER;

  byte_array_clear(result);
  for (size_t i = 0; i < token_count; i++) {
    uint8_t* token_bytes = NULL; size_t token_len = 0;
    // Try regular decoder first
    if (reverse_map_get(bpe->decoder, tokens[i], &token_bytes, &token_len)) {
      // Extend result array
      size_t new_len = result->len + token_len;
      uint8_t* new_bytes = (uint8_t*)realloc(result->bytes, new_len);
      if (!new_bytes) return ERROR_MEMORY_ALLOCATION;

      memcpy(new_bytes + result->len, token_bytes, token_len);
      result->bytes = new_bytes;
      result->len = new_len;
      continue;
    }

    // Try special tokens decoder
    if (bpe->special_tokens_decoder && reverse_map_get(bpe->special_tokens_decoder, tokens[i], &token_bytes, &token_len)) {
      size_t new_len = result->len + token_len; uint8_t* new_bytes = (uint8_t*)realloc(result->bytes, new_len);
      if (!new_bytes) return ERROR_MEMORY_ALLOCATION;

      memcpy(new_bytes + result->len, token_bytes, token_len);
      result->bytes = new_bytes;
      result->len = new_len;
      continue;
    }
    return ERROR_INVALID_TOKEN;
  }
  return OK;
}

// Decode single token to bytes
ShredError decode_single_token_bytes(CoreBPE* bpe, Rank token, ByteArray* result) {
  if (!bpe || !result) return ERROR_NULL_POINTER;
  byte_array_clear(result);
  uint8_t* token_bytes = NULL; size_t token_len = 0;

  // Try regular decoder first
  if (reverse_map_get(bpe->decoder, token, &token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len);
    if (!result->bytes) return ERROR_MEMORY_ALLOCATION;
    memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
    return OK;
  }

  // Try special tokens decoder
  if (bpe->special_tokens_decoder && reverse_map_get(bpe->special_tokens_decoder, token, &token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len);
    if (!result->bytes) return ERROR_MEMORY_ALLOCATION;
    memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
    return OK;
  }
  return ERROR_INVALID_TOKEN;
}

// Get total token count
size_t get_token_count(CoreBPE* bpe) {
  if (!bpe) return 0;
  size_t count = bpe->encoder ? bpe->encoder->size : 0;
  if (bpe->special_tokens_encoder) { count += bpe->special_tokens_encoder->size; }
  return count;
}

// Internal helper functions
static ShredError compile_regex(const char* pattern, regex_t* regex) {
  int result = regcomp(regex, pattern, REG_EXTENDED);
  return (result == 0) ? OK : ERROR_REGEX_COMPILE;
}

static ShredError find_regex_matches(regex_t* regex, const char* text, size_t** matches, size_t* match_count) {
  if (!regex || !text || !matches || !match_count) return ERROR_NULL_POINTER;

  size_t capacity = 16;
  *matches = (size_t*)malloc(capacity * sizeof(size_t));
  if (!*matches) return ERROR_MEMORY_ALLOCATION;

  *match_count = 0;
  regmatch_t match;
  const char* current = text;
  size_t offset = 0;

  while (regexec(regex, current, 1, &match, 0) == 0) {
    if (*match_count + 2 >= capacity) {
      capacity *= 2;
      size_t* new_matches = (size_t*)realloc(*matches, capacity * sizeof(size_t));
      if (!new_matches) {
        free(*matches);
        return ERROR_MEMORY_ALLOCATION;
      }
      *matches = new_matches;
    }
    (*matches)[(*match_count)++] = offset + match.rm_so;
    (*matches)[(*match_count)++] = offset + match.rm_eo;
    if (match.rm_eo == 0) break; // Avoid infinite loop on zero-length matches 
    current += match.rm_eo;
    offset += match.rm_eo;
  }
  return OK;
}

static ShredError byte_pair_encode_internal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result) {
  if (!piece || !encoder || !result) return ERROR_NULL_POINTER;
  if (piece_len == 0) return OK;
  if (piece_len == 1) {
    Rank token;
    if (hashmap_get(encoder, piece, 1, &token)) { return token_array_push(result, token); }
    return ERROR_INVALID_TOKEN;
  }

  size_t* parts = NULL;
  size_t parts_count = 0;
  ShredError err = byte_pair_merge(encoder, piece, piece_len, &parts, &parts_count);
  if (err != OK) return err;

  // Convert parts to tokens
  for (size_t i = 0; i < parts_count - 1; i++) {
    size_t start = parts[i]; size_t end = parts[i + 1]; size_t token_len = end - start;
    Rank token;
    if (!hashmap_get(encoder, piece + start, token_len, &token)) {
      free(parts);
      return ERROR_INVALID_TOKEN;
    } 
    err = token_array_push(result, token);
    if (err != OK) {
      free(parts);
      return err;
    }
  }
  free(parts);
  return OK;
}

static ShredError byte_pair_merge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count) {
  if (!ranks || !piece || !parts || !parts_count) return ERROR_NULL_POINTER;
  size_t capacity = piece_len + 2;
  *parts = (size_t*)malloc(capacity * sizeof(size_t));
  if (!*parts) return ERROR_MEMORY_ALLOCATION;
  *parts_count = 0;

  // Add all positions initially
  for (size_t i = 0; i < piece_len; i++) { (*parts)[(*parts_count)++] = i; }
  (*parts)[(*parts_count)++] = piece_len; // End marker
  if (piece_len < 2) return OK;
  // Find pairs and merge
  bool changed = true;
  while (changed && *parts_count > 2) {
    changed = false;
    Rank best_rank = C_UINT32_MAX;
    size_t best_idx = SIZE_MAX;

    // Find best pair to merge
    for (size_t i = 0; i < *parts_count - 2; i++) {
      size_t start1 = (*parts)[i]; size_t end1 = (*parts)[i + 1]; size_t end2 = (*parts)[i + 2];
      uint8_t pair[2] = {piece[start1], piece[end1]};
      Rank rank;
      if (hashmap_get(ranks, pair, 2, &rank) && rank < best_rank) {
        best_rank = rank;
        best_idx = i;
      }
    }

    // Perform merge if found
    if (best_idx != SIZE_MAX) {
      // Remove the middle part
      for (size_t i = best_idx + 1; i < *parts_count - 1; i++) { (*parts)[i] = (*parts)[i + 1]; }
      (*parts_count)--;
      changed = true;
    }
  }
  return OK;
}

ShredError get_token_byte_values(CoreBPE* bpe, ByteArray** results, size_t* count) {
  if (!bpe || !results || !count) return ERROR_NULL_POINTER;
  *count = 0;
  *results = NULL;
  if (!bpe->encoder) return ERROR_NULL_POINTER;

  // Count total tokens
  size_t total_tokens = bpe->encoder->size;
  if (bpe->special_tokens_encoder) { total_tokens += bpe->special_tokens_encoder->size; }
  if (total_tokens == 0) return OK;

  // Allocate result array
  *results = (ByteArray*)malloc(sizeof(ByteArray) * total_tokens);
  if (!*results) return ERROR_MEMORY_ALLOCATION;
  size_t result_idx = 0;

  // Process regular encoder
  for (size_t i = 0; i < bpe->encoder->bucket_count; i++) {
    HashMapNode* node = bpe->encoder->buckets[i];
    while (node && result_idx < total_tokens) {
      (*results)[result_idx].bytes = (uint8_t*)malloc(node->key_len);
      if (!(*results)[result_idx].bytes) {
        // Cleanup on error
        for (size_t j = 0; j < result_idx; j++) { free((*results)[j].bytes); }
        free(*results);
        *results = NULL;
        return ERROR_MEMORY_ALLOCATION;
      }
      memcpy((*results)[result_idx].bytes, node->key, node->key_len);
      (*results)[result_idx].len = node->key_len;
      result_idx++;
      node = node->next;
    }
  }

  // Process special tokens encoder
  if (bpe->special_tokens_encoder) {
    for (size_t i = 0; i < bpe->special_tokens_encoder->bucket_count; i++) {
      HashMapStrNode* node = bpe->special_tokens_encoder->buckets[i];
      while (node && result_idx < total_tokens) {
        size_t key_len = strlen(node->key);
        (*results)[result_idx].bytes = (uint8_t*)malloc(key_len);
        if (!(*results)[result_idx].bytes) {
          // Cleanup on error
          for (size_t j = 0; j < result_idx; j++) { free((*results)[j].bytes); }
          free(*results);
          *results = NULL;
          return ERROR_MEMORY_ALLOCATION;
        }
        memcpy((*results)[result_idx].bytes, node->key, key_len);
        (*results)[result_idx].len = key_len;
        result_idx++;
        node = node->next;
      }
    }
  }
  *count = result_idx;
  return OK;
}
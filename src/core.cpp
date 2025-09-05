#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <regex.h>
#include "hashmap.h"
#include "token.h"
#include "core.h"

#define C_UINT32_MAX 0xFFFFFFFF

static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count);
static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result);
static void compileRegex(const char* pattern, regex_t* regex);
static void findRegexMatches(regex_t* regex, const char* text, size_t** matches, size_t* match_count);

CoreBPE* shredCreate(uint8_t** encoder_keys, const size_t* encoder_key_lens, const Rank* encoder_values, size_t encoder_count, const char** special_token_keys, const Rank* special_token_values, size_t special_token_count, const char* pattern) {
  if (!encoder_keys || !encoder_key_lens || !encoder_values || !pattern) {
    fprintf(stderr, "SHRED>ERROR 101 <shredCreate() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  CoreBPE* bpe = (CoreBPE*)malloc(sizeof(CoreBPE));
  if (!bpe) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  memset(bpe, 0, sizeof(CoreBPE));
  bpe->encoder = hashmapCreate(encoder_count * 2);
  if (!bpe->encoder) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    shredFree(bpe);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < encoder_count; i++) hashmapInsert(bpe->encoder, encoder_keys[i], encoder_key_lens[i], encoder_values[i]);
  bpe->decoder = revmapCreate(encoder_count * 2);
  if (!bpe->decoder) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    shredFree(bpe);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < encoder_count; i++) revmapInsert(bpe->decoder, encoder_values[i], encoder_keys[i], encoder_key_lens[i]);
  if (special_token_keys && special_token_values && special_token_count > 0) {
    bpe->special_tokens_encoder = strmapCreate(special_token_count * 2);
    if (!bpe->special_tokens_encoder) {
      fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
      shredFree(bpe);
      exit(EXIT_FAILURE);
    }
    bpe->special_tokens_decoder = revmapCreate(special_token_count * 2);
    if (!bpe->special_tokens_decoder) {
      shredFree(bpe);
      return NULL;
    }
    for (size_t i = 0; i < special_token_count; i++) {
      strmapInsert(bpe->special_tokens_encoder, (char*)special_token_keys[i], special_token_values[i]);
      size_t key_len = strlen(special_token_keys[i]);
      revmapInsert(bpe->special_tokens_decoder, special_token_values[i], (const uint8_t*)special_token_keys[i], key_len);
    }
  }
  bpe->regex = malloc(sizeof(regex_t));
  if (!bpe->regex) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    shredFree(bpe);
    exit(EXIT_FAILURE);
  }
  compileRegex(pattern, (regex_t*)bpe->regex);
  return bpe;
}

void shredFree(CoreBPE* bpe) {
  if (!bpe) {
    fprintf(stderr, "SHRED>ERROR 101 <shredFree() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (bpe->encoder) { hashmapFree(bpe->encoder); }
  if (bpe->special_tokens_encoder) { strmapFree(bpe->special_tokens_encoder); }
  if (bpe->decoder) { revmapFree(bpe->decoder); }
  if (bpe->special_tokens_decoder) { revmapFree(bpe->special_tokens_decoder); }
  if (bpe->regex) {
    regfree((regex_t*)bpe->regex);
    free(bpe->regex);
  }
  if (bpe->special_regex) {
    regfree((regex_t*)bpe->special_regex);
    free(bpe->special_regex);
  }
  if (bpe->sorted_token_bytes) sortedTokensFree(bpe->sorted_token_bytes);
  free(bpe);
}

// Encode ordinary text (no special tokens)
void encodeOrdinary(CoreBPE* bpe, const char* text, TokenArray* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeOrdinary() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  tokenArrayClear(result);
  size_t* matches = NULL;
  size_t match_count = 0;
  findRegexMatches((regex_t*)bpe->regex, text, &matches, &match_count);
  for (size_t i = 0; i < match_count; i += 2) {
    size_t start = matches[i];
    size_t end = matches[i + 1];
    size_t piece_len = end - start;
    uint8_t* piece = (uint8_t*)(text + start);

    // trying direct lookup first else moving to BPE encoding
    Rank token;
    if (hashmapGet(bpe->encoder, piece, piece_len, &token)) { tokenArrayPush(result, token); }
    else { bytePairEncodeInternal(piece, piece_len, bpe->encoder, result); }
  }
  free(matches);
}

// Encode with special tokens support
void encode(CoreBPE* bpe, const char* text, char** allowed_special, size_t allowed_special_count, TokenArray* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encode() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  if (!allowed_special || allowed_special_count == 0) { return encodeOrdinary(bpe, text, result); } // if no special tokens are allowed, use encode_ordinary
  if (!bpe->special_tokens_encoder) { return encodeOrdinary(bpe, text, result); } // if no special tokens encoder is available, fall back to ordinary encoding
  tokenArrayClear(result);  
  const char* current = text;
  size_t text_len = strlen(text);
  while (current < text + text_len) {
    const char* next_special = NULL;
    size_t special_len = 0;
    Rank special_token = 0;
    for (size_t i = 0; i < allowed_special_count; i++) {    // checking for allowed special tokens at current position
      size_t token_len = strlen(allowed_special[i]);
      if (current + token_len <= text + text_len && 
          strncmp(current, allowed_special[i], token_len) == 0) {
        if (!next_special || current < next_special) {  // found a special token, check if it's the earliest one
          next_special = current;
          special_len = token_len;
          if (!strmapGet(bpe->special_tokens_encoder, allowed_special[i], &special_token)) continue;  // Special token not found in encoder, skip it
        }
      }
    }

    if (next_special == current) {  // encode the special token
      tokenArrayPush(result, special_token);
      current += special_len;
    } else {  // find the next special token in the remaining text
      const char* next_occurrence = NULL;
      size_t next_occurrence_len = 0;
      for (size_t i = 0; i < allowed_special_count; i++) {
        const char* found = strstr(current, allowed_special[i]);
        if (found && (!next_occurrence || found < next_occurrence)) {
          next_occurrence = found;
          next_occurrence_len = strlen(allowed_special[i]);
        }
      }

      // encode ordinary text up to the next special token (or end of string)
      const char* end_pos = next_occurrence ? next_occurrence : (text + text_len);
      size_t ordinary_len = end_pos - current;
      if (ordinary_len > 0) {
        char* ordinary_text = (char*)malloc(ordinary_len + 1);
        if (!ordinary_text) {
          fprintf(stderr, "SHRED>ERROR 102 <encode() in core.cpp>:  Couldn't allocate memory\n");
          exit(EXIT_FAILURE);
        }
        memcpy(ordinary_text, current, ordinary_len);
        ordinary_text[ordinary_len] = '\0';
        encodeOrdinary(bpe, ordinary_text, result);
        free(ordinary_text);
        current = end_pos;
      }
    }
  }
}

void encodeBytes(CoreBPE* bpe, const uint8_t* bytes, size_t byte_len, TokenArray* result) {
  if (!bpe || !bytes || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeBytes() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  tokenArrayClear(result);
  bytePairEncodeInternal(bytes, byte_len, bpe->encoder, result);
}

void encodeSingleToken(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, Rank* result) {
  if (!bpe || !piece || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeSingleToken() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (hashmapGet(bpe->encoder, (uint8_t*)piece, piece_len, result)) return; // try regular encoder first
  if (bpe->special_tokens_encoder) {  // try special tokens encoder if available
    char* piece_str = (char*)malloc(piece_len + 1); // converting to null-terminated string for special token lookup
    if (!piece_str) {
      fprintf(stderr, "SHRED>ERROR 102 <encodeSingleToken() in core.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    memcpy(piece_str, piece, piece_len);
    piece_str[piece_len] = '\0';
    bool found = strmapGet(bpe->special_tokens_encoder, piece_str, result);
    free(piece_str);
    if (found) return;
  }
}

void encodeSinglePiece(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, TokenArray* result) {
  if (!bpe || !piece || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeSinglePiece() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  tokenArrayClear(result);
  Rank token;   // trying direct lookup first
  if (hashmapGet(bpe->encoder, (uint8_t*)piece, piece_len, &token)) { return tokenArrayPush(result, token); }
  return bytePairEncodeInternal(piece, piece_len, bpe->encoder, result); // Use BPE encoding
}

// Decode tokens to bytes
void decodeBytes(CoreBPE* bpe, const Rank* tokens, size_t token_count, ByteArray* result) {
  if (!bpe || !tokens || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <decodeBytes() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  byteArrayClear(result);
  for (size_t i = 0; i < token_count; i++) {
    uint8_t* token_bytes = NULL; size_t token_len = 0;
    // trying regular decoder first
    if (revmapGet(bpe->decoder, tokens[i], &token_bytes, &token_len)) { // Extend result array
      size_t new_len = result->len + token_len;
      uint8_t* new_bytes = (uint8_t*)realloc(result->bytes, new_len);
      if (!new_bytes) {
        fprintf(stderr, "SHRED>ERROR 102 <decodeBytes() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      memcpy(new_bytes + result->len, token_bytes, token_len);
      result->bytes = new_bytes;
      result->len = new_len;
      continue;
    }

    // tryinh special tokens decoder
    if (bpe->special_tokens_decoder && revmapGet(bpe->special_tokens_decoder, tokens[i], &token_bytes, &token_len)) {
      size_t new_len = result->len + token_len; uint8_t* new_bytes = (uint8_t*)realloc(result->bytes, new_len);
      if (!new_bytes) {
        fprintf(stderr, "SHRED>ERROR 102 <decodeBytes() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }

      memcpy(new_bytes + result->len, token_bytes, token_len);
      result->bytes = new_bytes;
      result->len = new_len;
      continue;
    }
  }
}

// Decode single token to bytes
void decodeSingleTokenBytes(CoreBPE* bpe, Rank token, ByteArray* result) {
  if (!bpe || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <decodeSingleTokenBytes() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  byteArrayClear(result);
  uint8_t* token_bytes = NULL; size_t token_len = 0;

  // trying regular decoder first
  if (revmapGet(bpe->decoder, token, &token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len);
    if (!result->bytes) {
      fprintf(stderr, "SHRED>ERROR 102 <decodeSingleTokenBytes() in core.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
  }

  // trying special tokens decoder
  if (bpe->special_tokens_decoder && revmapGet(bpe->special_tokens_decoder, token, &token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len);
    if (!result->bytes) {
      fprintf(stderr, "SHRED>ERROR 102 <decodeSingleTokenBytes() in core.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
  }
}

size_t getTokenCount(CoreBPE* bpe) {
  if (!bpe) {
    fprintf(stderr, "SHRED>ERROR 101 <getTokenCount() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  size_t count = bpe->encoder ? bpe->encoder->size : 0;
  if (bpe->special_tokens_encoder) { count += bpe->special_tokens_encoder->size; }
  return count;
}

// Internal helper functions -----------------
static void compileRegex(const char* pattern, regex_t* regex) {
  int result = regcomp(regex, pattern, REG_EXTENDED);
  if (result == 0) {
    fprintf(stderr, "Couldn't compile the given Regex Pattern\n");
    exit(EXIT_FAILURE);
  }
}

static void findRegexMatches(regex_t* regex, const char* text, size_t** matches, size_t* match_count) {
  if (!regex || !text || !matches || !match_count) {
    fprintf(stderr, "SHRED>ERROR 101 <findRegexMatches() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  size_t capacity = 16;
  *matches = (size_t*)malloc(capacity * sizeof(size_t));
  if (!*matches) {
    fprintf(stderr, "SHRED>ERROR 102 <findRegexMatches() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }

  *match_count = 0;
  regmatch_t match;
  const char* current = text;
  size_t offset = 0;

  while (regexec(regex, current, 1, &match, 0) == 0) {
    if (*match_count + 2 >= capacity) {
      capacity *= 2;
      size_t* new_matches = (size_t*)realloc(*matches, capacity * sizeof(size_t));
      if (!new_matches) {
        fprintf(stderr, "SHRED>ERROR 102 <findRegexMatches() in core.cpp>:  Couldn't allocate memory\n");
        free(*matches);
        exit(EXIT_FAILURE);
      }
      *matches = new_matches;
    }
    (*matches)[(*match_count)++] = offset + match.rm_so;
    (*matches)[(*match_count)++] = offset + match.rm_eo;
    if (match.rm_eo == 0) break; // Avoid infinite loop on zero-length matches 
    current += match.rm_eo;
    offset += match.rm_eo;
  }

}

static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result) {
  if (!piece || !encoder || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <bytePairEncodeInternal() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (piece_len == 0) return;
  if (piece_len == 1) {
    Rank token;
    if (hashmapGet(encoder, (uint8_t*)piece, 1, &token)) { return tokenArrayPush(result, token); }
    fprintf(stderr, "SHRED>ERROR 202 <bytePairEncodeInternal() in core.cpp>  Encountered invalid Token\n");
  }

  size_t* parts = NULL;
  size_t parts_count = 0;
  bytePairMerge(encoder, piece, piece_len, &parts, &parts_count);
  for (size_t i = 0; i < parts_count - 1; i++) {
    size_t start = parts[i]; size_t end = parts[i + 1]; size_t token_len = end - start;
    Rank token;
    if (!hashmapGet(encoder, (uint8_t*)piece + start, token_len, &token)) {
      free(parts);
      fprintf(stderr, "SHRED>ERROR 202 <bytePairEncodeInternal() in core.cpp>  Encountered invalid Token\n");
    } 
    tokenArrayPush(result, token);
  }
  free(parts);
}

static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count) {
  if (!ranks || !piece || !parts || !parts_count) {
    fprintf(stderr, "SHRED>ERROR 101 <bytePairMerge() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  size_t capacity = piece_len + 2;
  *parts = (size_t*)malloc(capacity * sizeof(size_t));
  if (!*parts) {
    fprintf(stderr, "SHRED>ERROR 102 <bytePairMerge() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  *parts_count = 0;

  // adding all positions initially
  for (size_t i = 0; i < piece_len; i++) { (*parts)[(*parts_count)++] = i; }
  (*parts)[(*parts_count)++] = piece_len; // end marker
  if (piece_len < 2) return;
  // finding pairs and merge
  bool changed = true;
  while (changed && *parts_count > 2) {
    changed = false;
    Rank best_rank = C_UINT32_MAX;
    size_t best_idx = SIZE_MAX;

    // finding best pair to merge
    for (size_t i = 0; i < *parts_count - 2; i++) {
      size_t start1 = (*parts)[i]; size_t end1 = (*parts)[i + 1]; size_t end2 = (*parts)[i + 2];
      uint8_t pair[2] = {piece[start1], piece[end1]};
      Rank rank;
      if (hashmapGet(ranks, pair, 2, &rank) && rank < best_rank) {
        best_rank = rank;
        best_idx = i;
      }
    }

    if (best_idx != SIZE_MAX) {
      for (size_t i = best_idx + 1; i < *parts_count - 1; i++) { (*parts)[i] = (*parts)[i + 1]; }
      (*parts_count)--;
      changed = true;
    }
  }
}

void getTokenByteValues(CoreBPE* bpe, ByteArray** results, size_t* count) {
  if (!bpe || !results || !count) {
    fprintf(stderr, "SHRED>ERROR 101 <getTokenByteValues() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  *count = 0;
  *results = NULL;
  if (!bpe->encoder) {
    fprintf(stderr, "SHRED>ERROR 101 <getTokenByteValues() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  size_t total_tokens = bpe->encoder->size;
  if (bpe->special_tokens_encoder) { total_tokens += bpe->special_tokens_encoder->size; }
  if (total_tokens == 0) return;
  *results = (ByteArray*)malloc(sizeof(ByteArray) * total_tokens);
  if (!*results) {
    fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  size_t result_idx = 0;

  for (size_t i = 0; i < bpe->encoder->bucket_count; i++) {
    HashMapNode* node = bpe->encoder->buckets[i];
    while (node && result_idx < total_tokens) {
      (*results)[result_idx].bytes = (uint8_t*)malloc(node->key_len);
      if (!(*results)[result_idx].bytes) {

        for (size_t j = 0; j < result_idx; j++) { free((*results)[j].bytes); }
        free(*results);
        *results = NULL;
        fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      memcpy((*results)[result_idx].bytes, node->key, node->key_len);
      (*results)[result_idx].len = node->key_len;
      result_idx++;
      node = node->next;
    }
  }

  // processing special tokens encoder
  if (bpe->special_tokens_encoder) {
    for (size_t i = 0; i < bpe->special_tokens_encoder->bucket_count; i++) {
      HashMapStrNode* node = bpe->special_tokens_encoder->buckets[i];
      while (node && result_idx < total_tokens) {
        size_t key_len = strlen(node->key);
        (*results)[result_idx].bytes = (uint8_t*)malloc(key_len);
        if (!(*results)[result_idx].bytes) {

          for (size_t j = 0; j < result_idx; j++) { free((*results)[j].bytes); }
          free(*results);
          *results = NULL;
          fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
          exit(EXIT_FAILURE);
        }
        memcpy((*results)[result_idx].bytes, node->key, key_len);
        (*results)[result_idx].len = key_len;
        result_idx++;
        node = node->next;
      }
    }
  }
  *count = result_idx;
}
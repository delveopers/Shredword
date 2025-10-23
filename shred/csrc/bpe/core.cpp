// core.cpp  (patched)
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <regex>
#include <vector>
#include <string>
#include <string_view>

#include "hashmap.h"
#include "token.h"
#include "core.h"

#define C_UINT32_MAX 0xFFFFFFFFu
#define DECODE_BUFFER_INIT 4096

static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count);
static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result);
static void compileRegex(const char* pattern, std::regex** regex);
static void findRegexMatches(std::regex* regex, const char* text, size_t text_len, size_t** matches, size_t* match_count);

CoreBPE* shredCreate(uint8_t** encoder_keys, const size_t* encoder_key_lens, const Rank* encoder_values, size_t encoder_count, const char** special_token_keys, const Rank* special_token_values, size_t special_token_count, const char* pattern) {
  if (!encoder_keys || !encoder_key_lens || !encoder_values || !pattern) {
    fprintf(stderr, "SHRED>ERROR 101 <shredCreate() in core.cpp>:  Invalid or NULL Parameters\n");
    return nullptr;
  }
  CoreBPE* bpe = (CoreBPE*)malloc(sizeof(CoreBPE));
  if (!bpe) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    return nullptr;
  }
  memset(bpe, 0, sizeof(CoreBPE));

  bpe->encoder = hashmapCreate(encoder_count * 2);
  if (!bpe->encoder) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    shredFree(bpe);
    return nullptr;
  }
  for (size_t i = 0; i < encoder_count; ++i) { hashmapInsert(bpe->encoder, encoder_keys[i], encoder_key_lens[i], encoder_values[i]); }
  bpe->decoder = revmapCreate(encoder_count * 2);
  if (!bpe->decoder) {
    fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
    shredFree(bpe);
    return nullptr;
  }
  for (size_t i = 0; i < encoder_count; ++i) { revmapInsert(bpe->decoder, encoder_values[i], encoder_keys[i], encoder_key_lens[i]); }

  // optional special tokens
  if (special_token_keys && special_token_values && special_token_count > 0) {
    bpe->special_tokens_encoder = strmapCreate(special_token_count * 2);
    if (!bpe->special_tokens_encoder) {
      fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
      shredFree(bpe);
      return nullptr;
    }
    bpe->special_tokens_decoder = revmapCreate(special_token_count * 2);
    if (!bpe->special_tokens_decoder) {
      fprintf(stderr, "SHRED>ERROR 102 <shredCreate() in core.cpp>:  Couldn't allocate memory\n");
      shredFree(bpe);
      return nullptr;
    }
    for (size_t i = 0; i < special_token_count; ++i) {
      // strmapInsert copies the string internally (strdup)
      strmapInsert(bpe->special_tokens_encoder, special_token_keys[i], special_token_values[i]);
      size_t key_len = strlen(special_token_keys[i]);
      revmapInsert(bpe->special_tokens_decoder, special_token_values[i], (const uint8_t*)special_token_keys[i], key_len);
    }
  } else {
    bpe->special_tokens_encoder = nullptr;
    bpe->special_tokens_decoder = nullptr;
  }

  bpe->regex = nullptr;
  bpe->special_regex = nullptr;
  compileRegex(pattern, &bpe->regex);
  bpe->sorted_token_bytes = nullptr;  // sorted_token_bytes will be created by caller via getTokenByteValues (if needed)

  return bpe;
}

void shredFree(CoreBPE* bpe) {
  if (!bpe) return;

  if (bpe->encoder) { hashmapFree(bpe->encoder); bpe->encoder = nullptr; }
  if (bpe->special_tokens_encoder) { strmapFree(bpe->special_tokens_encoder); bpe->special_tokens_encoder = nullptr; }
  if (bpe->decoder) { revmapFree(bpe->decoder); bpe->decoder = nullptr; }
  if (bpe->special_tokens_decoder) { revmapFree(bpe->special_tokens_decoder); bpe->special_tokens_decoder = nullptr; }
  if (bpe->regex) { delete bpe->regex; bpe->regex = nullptr; }
  if (bpe->special_regex) { delete bpe->special_regex; bpe->special_regex = nullptr; }
  if (bpe->sorted_token_bytes) { sortedTokensFree(bpe->sorted_token_bytes); bpe->sorted_token_bytes = nullptr; }

  free(bpe);
}

static void findRegexMatches(std::regex* regex, const char* text, size_t text_len, size_t** matches, size_t* match_count) {
  if (!regex || !text || !matches || !match_count) {
    fprintf(stderr, "SHRED>ERROR <findRegexMatches()>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  std::string s(text, text_len);
  std::vector<size_t> tmp;
  tmp.reserve(64);

  try {
    std::sregex_iterator it(s.begin(), s.end(), *regex);
    std::sregex_iterator end;
    for (; it != end; ++it) {
      std::smatch m = *it;
      tmp.push_back((size_t)m.position(0));
      tmp.push_back((size_t)(m.position(0) + m.length(0)));
    }
  } catch (const std::regex_error& e) {
    fprintf(stderr, "SHRED>ERROR <findRegexMatches()>: regex iteration error: %s\n", e.what());
    *matches = nullptr;
    *match_count = 0;
    return;
  }

  *match_count = tmp.size();
  if (*match_count > 0) {
    *matches = (size_t*)malloc(tmp.size() * sizeof(size_t));
    if (!*matches) {
      fprintf(stderr, "SHRED>ERROR <findRegexMatches()>: Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    memcpy(*matches, tmp.data(), tmp.size() * sizeof(size_t));
  } else { *matches = nullptr; }
}

void encodeOrdinary(CoreBPE* bpe, const char* text, TokenArray* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeOrdinary() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  tokenArrayClear(result);
  size_t* matches = nullptr, match_count = 0, text_len = strlen(text);
  findRegexMatches(bpe->regex, text, text_len, &matches, &match_count);
  for (size_t i = 0; i + 1 < match_count; i += 2) {
    size_t start = matches[i], end = matches[i + 1];
    size_t piece_len = end - start;
    const uint8_t* piece = (const uint8_t*)(text + start);
    Rank token;
    if (hashmapGet(bpe->encoder, piece, piece_len, &token)) { tokenArrayPush(result, token); } else { bytePairEncodeInternal(piece, piece_len, bpe->encoder, result); }
  }
  free(matches);
}

void encode(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, TokenArray* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encode() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  // if allowed_special is NULL or count 0, fall back to ordinary encode
  if (!allowed_special || allowed_special_count == 0) {
    encodeOrdinary(bpe, text, result);
    return;
  }
  if (!bpe->special_tokens_encoder) {
    encodeOrdinary(bpe, text, result);
    return;
  }
  tokenArrayClear(result);
  std::vector<size_t> special_lens(allowed_special_count);
  std::vector<Rank> special_ranks(allowed_special_count);
  for (size_t i = 0; i < allowed_special_count; ++i) {
    special_lens[i] = strlen(allowed_special[i]);
    Rank r = 0;
    bool found = strmapGet(bpe->special_tokens_encoder, allowed_special[i], &r);
    special_ranks[i] = found ? r : (Rank)C_UINT32_MAX;
  }
  const char* current = text;
  size_t text_len = strlen(text);
  const char* text_end = text + text_len;
  while (current < text_end) {      // First check immediate position for a special match (greedy)
    bool matched_special = false;
    for (size_t i = 0; i < allowed_special_count; ++i) {
      size_t slen = special_lens[i];
      if (slen == 0) continue;
      if ((size_t)(text_end - current) >= slen && memcmp(current, allowed_special[i], slen) == 0) {
        Rank sr = special_ranks[i];
        if (sr != (Rank)C_UINT32_MAX) {
          tokenArrayPush(result, sr);
          current += slen;
          matched_special = true;
          break;
        }
      } // if special token not present in encoder/strmap, treat as ordinary - skip it here
    }
    if (matched_special) continue;

    // find next occurrence of any allowed_special after current
    const char* next_occurrence = nullptr;
    size_t next_len = 0;
    for (size_t i = 0; i < allowed_special_count; ++i) {
      const char* found = strstr(current, allowed_special[i]);
      if (found) {
        if (!next_occurrence || found < next_occurrence) {
          next_occurrence = found;
          next_len = special_lens[i];
        }
      }
    }

    const char* end_pos = next_occurrence ? next_occurrence : text_end;
    size_t ordinary_len = (size_t)(end_pos - current);
    if (ordinary_len == 0) {  // no ordinary text; advance (shouldn't happen because matched_special handled immediate match)
      current = end_pos;
      continue;
    }
    size_t* matches = nullptr;  // processing ordinary substring [current, end_pos) without mutating input
    size_t match_count = 0;
    findRegexMatches(bpe->regex, current, ordinary_len, &matches, &match_count);

    TokenArray* temp_result = tokenArrayCreate(128);
    if (!temp_result) {
      fprintf(stderr, "SHRED>ERROR 102 <encode() in core.cpp>: Couldn't allocate temporary token array\n");
      exit(EXIT_FAILURE);
    }

    for (size_t m = 0; m + 1 < match_count; m += 2) {
      size_t start = matches[m];
      size_t end = matches[m + 1];
      // clamp to substring bounds (should already be within ordinary_len)
      if (start >= ordinary_len) break;
      if (end > ordinary_len) end = ordinary_len;
      size_t piece_len = end - start;
      const uint8_t* piece = (const uint8_t*)(current + start);
      Rank token;
      if (hashmapGet(bpe->encoder, piece, piece_len, &token)) { tokenArrayPush(temp_result, token); } else { bytePairEncodeInternal(piece, piece_len, bpe->encoder, temp_result); }
    }
    // append temp_result tokens to result
    for (size_t k = 0; k < temp_result->count; ++k) tokenArrayPush(result, temp_result->tokens[k]);
    tokenArrayFree(temp_result);
    free(matches);
    current = end_pos;
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
  if (hashmapGet(bpe->encoder, (uint8_t*)piece, piece_len, result)) return;
  if (bpe->special_tokens_encoder) {
    std::string tmp((const char*)piece, piece_len);
    Rank r = 0;
    if (strmapGet(bpe->special_tokens_encoder, tmp.c_str(), &r)) {
      *result = r;
      return;
    }
  }  // if not found, caller will inspect return value (no change)
}

// encode single lexical piece (possibly multi-byte) -> token(s)
void encodeSinglePiece(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, TokenArray* result) {
  if (!bpe || !piece || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeSinglePiece() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  tokenArrayClear(result);
  Rank token;
  if (hashmapGet(bpe->encoder, (uint8_t*)piece, piece_len, &token)) {
    tokenArrayPush(result, token);
    return;
  }
  bytePairEncodeInternal(piece, piece_len, bpe->encoder, result);
}

void decodeBytes(CoreBPE* bpe, const Rank* tokens, size_t token_count, ByteArray* result) {
  if (!bpe || !tokens || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <decodeBytes() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  // free previous buffer if present to avoid leaking
  if (result->bytes) { free(result->bytes); result->bytes = nullptr; result->len = 0; }
  if (token_count == 0) {
    result->bytes = nullptr;
    result->len = 0;
    return;
  }

  size_t capacity = DECODE_BUFFER_INIT;
  uint8_t* bytes = (uint8_t*)malloc(capacity);
  if (!bytes) {
    fprintf(stderr, "SHRED>ERROR 102 <decodeBytes() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  size_t len = 0;

  for (size_t i = 0; i < token_count; ++i) {
    const uint8_t* token_bytes = nullptr; size_t token_len = 0;
    if (revmapGet(bpe->decoder, tokens[i], (uint8_t**)&token_bytes, &token_len)) {
      while (len + token_len > capacity) capacity *= 2;
      uint8_t* newb = (uint8_t*)realloc(bytes, capacity);
      if (!newb) {
        free(bytes);
        fprintf(stderr, "SHRED>ERROR 102 <decodeBytes() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      bytes = newb;
      memcpy(bytes + len, token_bytes, token_len);
      len += token_len;
      continue;
    }
    if (bpe->special_tokens_decoder && revmapGet(bpe->special_tokens_decoder, tokens[i], (uint8_t**)&token_bytes, &token_len)) {
      while (len + token_len > capacity) capacity *= 2;
      uint8_t* newb = (uint8_t*)realloc(bytes, capacity);
      if (!newb) {
        free(bytes);
        fprintf(stderr, "SHRED>ERROR 102 <decodeBytes() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      bytes = newb;
      memcpy(bytes + len, token_bytes, token_len);
      len += token_len;
    }
  }
  result->bytes = bytes;
  result->len = len;
}

// decode single token -> bytes (allocates result->bytes; caller must free or byteArrayClear)
void decodeSingleTokenBytes(CoreBPE* bpe, Rank token, ByteArray* result) {
  if (!bpe || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <decodeSingleTokenBytes() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (result->bytes) { free(result->bytes); result->bytes = nullptr; result->len = 0; }

  const uint8_t* token_bytes = nullptr; size_t token_len = 0;
  if (revmapGet(bpe->decoder, token, (uint8_t**)&token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len ? token_len : 1);
    if (!result->bytes) {
      fprintf(stderr, "SHRED>ERROR 102 <decodeSingleTokenBytes() in core.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    if (token_len) memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
    return;
  }
  if (bpe->special_tokens_decoder && revmapGet(bpe->special_tokens_decoder, token, (uint8_t**)&token_bytes, &token_len)) {
    result->bytes = (uint8_t*)malloc(token_len ? token_len : 1);
    if (!result->bytes) {
      fprintf(stderr, "SHRED>ERROR 102 <decodeSingleTokenBytes() in core.cpp>:  Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    if (token_len) memcpy(result->bytes, token_bytes, token_len);
    result->len = token_len;
  }
}

// token count: encoder + special encoder
size_t getTokenCount(CoreBPE* bpe) {
  if (!bpe) {
    fprintf(stderr, "SHRED>ERROR 101 <getTokenCount() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  size_t count = bpe->encoder ? bpe->encoder->size : 0;
  if (bpe->special_tokens_encoder) count += bpe->special_tokens_encoder->size;
  return count;
}

// compileRegex: keep compiled regex on bpe, fallback if pattern invalid
static void compileRegex(const char* pattern, std::regex** regex) {
  if (!pattern || !regex) return;
  try {
    *regex = new std::regex(pattern, std::regex::ECMAScript | std::regex::optimize);
  } catch (const std::regex_error& e) {
    const char* fallback_pattern = "[A-Za-z]+|[0-9]+|[^A-Za-z0-9\\s]+|\\s+";
    fprintf(stderr, "SHRED>WARNING <compileRegex()>: Regex failed (%s), using fallback\n", e.what());
    try {
      *regex = new std::regex(fallback_pattern, std::regex::ECMAScript | std::regex::optimize);
    } catch (...) {
      *regex = nullptr;
    }
  }
}

// Byte pair encode internal: produce a sequence of token ranks for a byte sequence
static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result) {
  if (!piece || !encoder || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <bytePairEncodeInternal() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (piece_len == 0) return;
  if (piece_len == 1) {
    Rank token;
    if (hashmapGet(encoder, (uint8_t*)piece, 1, &token)) { tokenArrayPush(result, token); return; }
    tokenArrayPush(result, 0);
    return;
  }

  size_t* parts = nullptr;
  size_t parts_count = 0;
  bytePairMerge(encoder, piece, piece_len, &parts, &parts_count);
  if (!parts || parts_count < 2) {
    // fallback: emit single-byte tokens
    for (size_t i = 0; i < piece_len; ++i) {
      Rank token;
      if (hashmapGet(encoder, (uint8_t*)(piece + i), 1, &token)) tokenArrayPush(result, token);
      else tokenArrayPush(result, 0);
    }
    free(parts);
    return;
  }

  for (size_t idx = 0; idx + 1 < parts_count; ++idx) {
    size_t start = parts[idx];
    size_t end = parts[idx + 1];
    size_t token_len = end - start;
    Rank token;
    if (hashmapGet(encoder, (uint8_t*)(piece + start), token_len, &token)) {
      tokenArrayPush(result, token);
    } else {
      // split into single bytes if no multi-byte token
      for (size_t j = start; j < end; ++j) {
        if (hashmapGet(encoder, (uint8_t*)(piece + j), 1, &token)) tokenArrayPush(result, token);
        else tokenArrayPush(result, 0);
      }
    }
  }

  free(parts);
}

// bytePairMerge: greedily merge adjacent parts while there is a better-ranked pair to merge
static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count) {
  if (!ranks || !piece || !parts || !parts_count) {
    fprintf(stderr, "SHRED>ERROR 101 <bytePairMerge() in core.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (piece_len == 0) {
    *parts = nullptr;
    *parts_count = 0;
    return;
  }

  // initial parts: 0,1,2,...,piece_len
  size_t capacity = piece_len + 1;
  *parts = (size_t*)malloc(sizeof(size_t) * capacity);
  if (!*parts) {
    fprintf(stderr, "SHRED>ERROR 102 <bytePairMerge() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  *parts_count = 0;
  for (size_t i = 0; i <= piece_len; ++i) (*parts)[(*parts_count)++] = i;
  if (piece_len < 2) return;

  bool changed = true;
  while (changed && *parts_count > 2) {
    changed = false;
    Rank best_rank = (Rank)C_UINT32_MAX;
    size_t best_idx = SIZE_MAX;
    // check adjacent pairs: combine start at parts[i] to parts[i+2] (two adjacent tokens combined)
    for (size_t i = 0; i + 2 < *parts_count; ++i) {
      size_t start1 = (*parts)[i];
      size_t end1 = (*parts)[i + 1];
      size_t end2 = (*parts)[i + 2];
      size_t pair_len = end2 - start1;
      Rank rank = 0;
      if (pair_len > 0 && hashmapGet(ranks, (uint8_t*)(piece + start1), pair_len, &rank)) {
        if (rank < best_rank) {
          best_rank = rank;
          best_idx = i + 1; // index of the boundary to remove
        }
      }
    }
    if (best_idx != SIZE_MAX) {
      for (size_t i = best_idx; i + 1 < *parts_count; ++i) { (*parts)[i] = (*parts)[i + 1]; }
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
  *results = nullptr;
  *count = 0;
  if (!bpe->encoder) return;
  size_t total_tokens = bpe->encoder->size;
  if (bpe->special_tokens_encoder) total_tokens += bpe->special_tokens_encoder->size;
  if (total_tokens == 0) return;

  ByteArray* arr = (ByteArray*)malloc(sizeof(ByteArray) * total_tokens);
  if (!arr) {
    fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  size_t idx = 0;
  for (size_t i = 0; i < bpe->encoder->bucket_count && idx < total_tokens; ++i) {
    HashMapNode* node = bpe->encoder->buckets[i];
    while (node && idx < total_tokens) {
      size_t len = node->key_len;
      arr[idx].bytes = (uint8_t*)malloc(len ? len : 1);
      if (!arr[idx].bytes) {
        for (size_t j = 0; j < idx; ++j) free(arr[j].bytes);
        free(arr);
        fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      if (len) memcpy(arr[idx].bytes, node->key, len);
      arr[idx].len = len;
      idx++;
      node = node->next;
    }
  }

  if (bpe->special_tokens_encoder) {
    for (size_t i = 0; i < bpe->special_tokens_encoder->bucket_count && idx < total_tokens; ++i) {
      HashMapStrNode* node = bpe->special_tokens_encoder->buckets[i];
      while (node && idx < total_tokens) {
        size_t key_len = strlen(node->key);
        arr[idx].bytes = (uint8_t*)malloc(key_len ? key_len : 1);
        if (!arr[idx].bytes) {
          for (size_t j = 0; j < idx; ++j) free(arr[j].bytes);
          free(arr);
          fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>:  Couldn't allocate memory\n");
          exit(EXIT_FAILURE);
        }
        if (key_len) memcpy(arr[idx].bytes, node->key, key_len);
        arr[idx].len = key_len;
        idx++;
        node = node->next;
      }
    }
  }

  *results = arr;
  *count = idx;
}
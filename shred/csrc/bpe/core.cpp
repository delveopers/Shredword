#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <regex>
#include <vector>
#include <string_view>

#include "hashmap.h"
#include "token.h"
#include "core.h"

#define C_UINT32_MAX 0xFFFFFFFFu
#define DECODE_BUFFER_INIT 4096

static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count);
static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result);
static void compileRegex(const char* pattern, std::regex** regex);
static void findRegexMatches(std::regex* regex, std::string_view text, size_t** matches, size_t* match_count);

CoreBPE* shredCreate(uint8_t** encoder_keys, const size_t* encoder_key_lens, const Rank* encoder_values, size_t encoder_count, const char** special_token_keys, const Rank* special_token_values, size_t special_token_count, const char* pattern) {
  if (!encoder_keys || !encoder_key_lens || !encoder_values || !pattern) return nullptr;

  CoreBPE* bpe = (CoreBPE*)calloc(1, sizeof(CoreBPE));
  if (!bpe) return nullptr;

  bpe->encoder = hashmapCreate(encoder_count * 2);
  for (size_t i = 0; i < encoder_count; ++i) {
    hashmapInsert(bpe->encoder, encoder_keys[i], encoder_key_lens[i], encoder_values[i]);
  }

  bpe->decoder = revmapCreate(encoder_count * 2);
  for (size_t i = 0; i < encoder_count; ++i) {
    revmapInsert(bpe->decoder, encoder_values[i], encoder_keys[i], encoder_key_lens[i]);
  }

  if (special_token_keys && special_token_values && special_token_count) {
    bpe->special_tokens_encoder = strmapCreate(special_token_count * 2);
    bpe->special_tokens_decoder = revmapCreate(special_token_count * 2);
    for (size_t i = 0; i < special_token_count; ++i) {
      strmapInsert(bpe->special_tokens_encoder, special_token_keys[i], special_token_values[i]);
      size_t len = strlen(special_token_keys[i]);
      revmapInsert(
        bpe->special_tokens_decoder,
        special_token_values[i],
        (const uint8_t*)special_token_keys[i],
        len
      );
    }
  }

  compileRegex(pattern, &bpe->regex);
  return bpe;
}

void shredFree(CoreBPE* bpe) {
  if (!bpe) return;
  hashmapFree(bpe->encoder);
  strmapFree(bpe->special_tokens_encoder);
  revmapFree(bpe->decoder);
  revmapFree(bpe->special_tokens_decoder);
  delete bpe->regex;
  delete bpe->special_regex;
  sortedTokensFree(bpe->sorted_token_bytes);
  free(bpe);
}

static void compileRegex(const char* pattern, std::regex** regex) {
  try {
    *regex = new std::regex(pattern, std::regex::ECMAScript | std::regex::optimize);
  } catch (...) {
    *regex = new std::regex("[A-Za-z]+|[0-9]+|[^A-Za-z0-9\\s]+|\\s+", std::regex::ECMAScript | std::regex::optimize);
  }
}

static void findRegexMatches(std::regex* regex, std::string_view text, size_t** matches, size_t* match_count) {
  std::vector<size_t> tmp;
  tmp.reserve(64);

  const char* base = text.data();
  auto begin = std::cregex_iterator(base, base + text.size(), *regex);
  auto end = std::cregex_iterator();

  for (auto it = begin; it != end; ++it) {
    tmp.push_back(it->position());
    tmp.push_back(it->position() + it->length());
  }

  *match_count = tmp.size();
  if (!tmp.empty()) {
    *matches = (size_t*)malloc(tmp.size() * sizeof(size_t));
    memcpy(*matches, tmp.data(), tmp.size() * sizeof(size_t));
  } else {
    *matches = nullptr;
  }
}

void encodeOrdinary(CoreBPE* bpe, const char* text, TokenArray* result) {
  tokenArrayClear(result);
  size_t* matches = nullptr;
  size_t match_count = 0;
  std::string_view view(text, strlen(text));

  findRegexMatches(bpe->regex, view, &matches, &match_count);
  for (size_t i = 0; i + 1 < match_count; i += 2) {
    size_t start = matches[i];
    size_t len = matches[i + 1] - start;
    const uint8_t* piece = (const uint8_t*)(text + start);
    Rank token;
    if (hashmapGet(bpe->encoder, piece, len, &token)) tokenArrayPush(result, token);
    else bytePairEncodeInternal(piece, len, bpe->encoder, result);
  }
  free(matches);
}

static void bytePairEncodeInternal(const uint8_t* piece, size_t piece_len, HashMap* encoder, TokenArray* result) {
  if (piece_len == 0) return;

  Rank token;
  if (hashmapGet(encoder, piece, piece_len, &token)) {
    tokenArrayPush(result, token);
    return;
  }

  if (piece_len == 1) {
    tokenArrayPush(result, 0);
    return;
  }

  size_t* parts = nullptr;
  size_t parts_count = 0;
  bytePairMerge(encoder, piece, piece_len, &parts, &parts_count);

  for (size_t i = 0; i + 1 < parts_count; ++i) {
    size_t start = parts[i];
    size_t len = parts[i + 1] - start;
    if (hashmapGet(encoder, piece + start, len, &token)) {
      tokenArrayPush(result, token);
    } else {
      for (size_t j = 0; j < len; ++j) tokenArrayPush(result, 0);
    }
  }
  free(parts);
}

static void bytePairMerge(HashMap* ranks, const uint8_t* piece, size_t piece_len, size_t** parts, size_t* parts_count) {
  *parts_count = piece_len + 1;
  *parts = (size_t*)malloc((*parts_count) * sizeof(size_t));
  for (size_t i = 0; i <= piece_len; ++i) (*parts)[i] = i;
  if (piece_len < 2) return;
  while (true) {
    Rank best = C_UINT32_MAX;
    size_t best_idx = SIZE_MAX;

    for (size_t i = 0; i + 2 < *parts_count; ++i) {
      size_t s = (*parts)[i];
      size_t e = (*parts)[i + 2];
      Rank r;
      if (hashmapGet(ranks, piece + s, e - s, &r) && r < best) {
        best = r;
        best_idx = i + 1;
      }
    }

    if (best_idx == SIZE_MAX) break;

    memmove(
      (*parts) + best_idx,
      (*parts) + best_idx + 1,
      (*parts_count - best_idx - 1) * sizeof(size_t)
    );
    (*parts_count)--;
  }
}

void encodeBytes(CoreBPE* bpe, const uint8_t* bytes, size_t len, TokenArray* result) {
  tokenArrayClear(result);
  bytePairEncodeInternal(bytes, len, bpe->encoder, result);
}

void decodeBytes(CoreBPE* bpe, const Rank* tokens, size_t count, ByteArray* result) {
  free(result->bytes);
  result->bytes = nullptr;
  result->len = 0;

  size_t cap = DECODE_BUFFER_INIT;
  uint8_t* buf = (uint8_t*)malloc(cap);
  size_t len = 0;

  for (size_t i = 0; i < count; ++i) {
    uint8_t* val;
    size_t vlen;
    if (!revmapGet(bpe->decoder, tokens[i], &val, &vlen)) continue;
    if (len + vlen > cap) {
      while (len + vlen > cap) cap <<= 1;
      buf = (uint8_t*)realloc(buf, cap);
    }
    memcpy(buf + len, val, vlen);
    len += vlen;
  }
  result->bytes = buf;
  result->len = len;
}

size_t getTokenCount(CoreBPE* bpe) {
  size_t n = bpe->encoder ? bpe->encoder->size : 0;
  if (bpe->special_tokens_encoder) n += bpe->special_tokens_encoder->size;
  return n;
}

void encode(CoreBPE* bpe, const char* text, const char** allowed_special, size_t allowed_special_count, TokenArray* result) {
  if (!bpe || !text || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encode() in core.cpp>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  if (!allowed_special || allowed_special_count == 0 || !bpe->special_tokens_encoder) {
    encodeOrdinary(bpe, text, result);
    return;
  }
  tokenArrayClear(result);
  std::vector<size_t> special_lens(allowed_special_count);
  std::vector<Rank> special_ranks(allowed_special_count);

  for (size_t i = 0; i < allowed_special_count; ++i) {
    special_lens[i] = strlen(allowed_special[i]);
    Rank r = 0;
    if (strmapGet(bpe->special_tokens_encoder, allowed_special[i], &r)) special_ranks[i] = r;
    else special_ranks[i] = (Rank)C_UINT32_MAX;
  }

  const char* cur = text;
  const char* end = text + strlen(text);

  while (cur < end) {
    bool matched = false;
    for (size_t i = 0; i < allowed_special_count; ++i) {
      size_t len = special_lens[i];
      if (len == 0) continue;
      if ((size_t)(end - cur) >= len &&
          memcmp(cur, allowed_special[i], len) == 0 &&
          special_ranks[i] != (Rank)C_UINT32_MAX) {
        tokenArrayPush(result, special_ranks[i]);
        cur += len;
        matched = true;
        break;
      }
    }

    if (matched) continue;

    const char* next_special = end;
    for (size_t i = 0; i < allowed_special_count; ++i) {
      const char* pos = strstr(cur, allowed_special[i]);
      if (pos && pos < next_special) next_special = pos;
    }
    size_t span_len = (size_t)(next_special - cur);
    if (span_len == 0) {
      cur = next_special;
      continue;
    }
    TokenArray* tmp = tokenArrayCreate(128);
    if (!tmp) {
      fprintf(stderr, "SHRED>ERROR 102 <encode() in core.cpp>: Allocation failed\n");
      exit(EXIT_FAILURE);
    }
    encodeOrdinary(bpe, cur, tmp);
    for (size_t i = 0; i < tmp->count; ++i) tokenArrayPush(result, tmp->tokens[i]);
    tokenArrayFree(tmp);
    cur = next_special;
  }
}

void encodeSingleToken(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, Rank* result) {
  if (!bpe || !piece || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeSingleToken() in core.cpp>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (hashmapGet(bpe->encoder, piece, piece_len, result)) return;
  if (bpe->special_tokens_encoder) {
    std::string_view sv((const char*)piece, piece_len);
    Rank r = 0;
    if (strmapGet(bpe->special_tokens_encoder, std::string(sv).c_str(), &r)) {
      *result = r;
      return;
    }
  }
}

void encodeSinglePiece(CoreBPE* bpe, const uint8_t* piece, size_t piece_len, TokenArray* result) {
  if (!bpe || !piece || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <encodeSinglePiece() in core.cpp>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  tokenArrayClear(result);
  Rank token;
  if (hashmapGet(bpe->encoder, piece, piece_len, &token)) {
    tokenArrayPush(result, token);
    return;
  }
  bytePairEncodeInternal(piece, piece_len, bpe->encoder, result);
}

void decodeSingleTokenBytes(CoreBPE* bpe, Rank token, ByteArray* result) {
  if (!bpe || !result) {
    fprintf(stderr, "SHRED>ERROR 101 <decodeSingleTokenBytes() in core.cpp>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  free(result->bytes);
  result->bytes = nullptr;
  result->len = 0;

  const uint8_t* bytes = nullptr;
  size_t len = 0;

  if (revmapGet(bpe->decoder, token, (uint8_t**)&bytes, &len) || (bpe->special_tokens_decoder && revmapGet(bpe->special_tokens_decoder, token, (uint8_t**)&bytes, &len))) {
    result->bytes = (uint8_t*)malloc(len ? len : 1);
    if (!result->bytes) {
      fprintf(stderr, "SHRED>ERROR 102 <decodeSingleTokenBytes() in core.cpp>: Couldn't allocate memory\n");
      exit(EXIT_FAILURE);
    }
    if (len) memcpy(result->bytes, bytes, len);
    result->len = len;
  }
}

void getTokenByteValues(CoreBPE* bpe, ByteArray** results, size_t* count) {
  if (!bpe || !results || !count) {
    fprintf(stderr, "SHRED>ERROR 101 <getTokenByteValues() in core.cpp>: Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  *results = nullptr;
  *count = 0;

  size_t total = bpe->encoder ? bpe->encoder->size : 0;
  if (bpe->special_tokens_encoder) total += bpe->special_tokens_encoder->size;
  if (total == 0) return;

  ByteArray* arr = (ByteArray*)malloc(sizeof(ByteArray) * total);
  if (!arr) {
    fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>: Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  size_t idx = 0;
  for (size_t i = 0; i < bpe->encoder->bucket_count; ++i) {
    HashMapNode* node = bpe->encoder->buckets[i];
    while (node) {
      arr[idx].len = node->key_len;
      arr[idx].bytes = (uint8_t*)malloc(node->key_len ? node->key_len : 1);
      if (!arr[idx].bytes) {
        for (size_t j = 0; j < idx; ++j) free(arr[j].bytes);
        free(arr);
        fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>: Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      if (node->key_len) memcpy(arr[idx].bytes, node->key, node->key_len);
      idx++;
      node = node->next;
    }
  }

  if (bpe->special_tokens_encoder) {
    for (size_t i = 0; i < bpe->special_tokens_encoder->bucket_count; ++i) {
      HashMapStrNode* node = bpe->special_tokens_encoder->buckets[i];
      while (node) {
        size_t len = strlen(node->key);
        arr[idx].len = len;
        arr[idx].bytes = (uint8_t*)malloc(len ? len : 1);
        if (!arr[idx].bytes) {
          for (size_t j = 0; j < idx; ++j) free(arr[j].bytes);
          free(arr);
          fprintf(stderr, "SHRED>ERROR 102 <getTokenByteValues() in core.cpp>: Couldn't allocate memory\n");
          exit(EXIT_FAILURE);
        }
        if (len) memcpy(arr[idx].bytes, node->key, len);
        idx++;
        node = node->next;
      }
    }
  }
  *results = arr;
  *count = idx;
}
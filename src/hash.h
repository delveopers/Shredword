/**
  @file hash.h
  @brief Hash map implementations for storing encoder/decoder mappings in BPE tokenization

  * This file provides hash map data structures and functions for:
  * - Regular hash maps (byte keys -> rank values) for token encoding
  * - String hash maps (string keys -> rank values) for special token encoding  
  * - Reverse maps (rank keys -> byte values) for token decoding
  * Used internally by the CoreBPE tokenizer for fast token lookup and conversion.
*/

#ifndef __HASH__H__
#define __HASH__H__

#include <stddef.h>
#include <stdint.h>

// Type definitions
typedef uint32_t Rank;

// Hash map structures
typedef struct HashMapNode {
  uint8_t* key;
  size_t key_len;
  Rank value;
  struct HashMapNode* next;
} HashMapNode;

typedef struct {
  HashMapNode** buckets;
  size_t bucket_count;
  size_t size;
} HashMap;

typedef struct HashMapStrNode {
  char* key;
  Rank value;
  struct HashMapStrNode* next;
} HashMapStrNode;

typedef struct {
  HashMapStrNode** buckets;
  size_t bucket_count;
  size_t size;
} HashMapStr;

// Reverse hash map for decoding
typedef struct ReverseMapNode {
  Rank key;
  uint8_t* value;
  size_t value_len;
  struct ReverseMapNode* next;
} ReverseMapNode;

typedef struct {
  ReverseMapNode** buckets;
  size_t bucket_count;
  size_t size;
} ReverseMap;

typedef enum {
  OK = 0,
  ERROR_NULL_POINTER = -1,
  ERROR_MEMORY_ALLOCATION = -2,
  ERROR_INVALID_TOKEN = -3,
  ERROR_REGEX_COMPILE = -4,
  ERROR_REGEX_MATCH = -5,
  ERROR_INVALID_UTF8 = -6
} ShredError;

extern "C" {
  HashMap* hashmap_new(size_t bucket_count);
  void hashmap_free(HashMap* map);
  bool hashmap_get(HashMap* map, const uint8_t* key, size_t key_len, Rank* value);
  ShredError hashmap_insert(HashMap* map, const uint8_t* key, size_t key_len, Rank value);
  HashMapStr* hashmap_str_new(size_t bucket_count);
  void hashmap_str_free(HashMapStr* map);
  bool hashmap_str_get(HashMapStr* map, const char* key, Rank* value);
  ReverseMap* reverse_map_new(size_t bucket_count);
  void reverse_map_free(ReverseMap* map);
  bool reverse_map_get(ReverseMap* map, Rank key, uint8_t** value, size_t* value_len);
  ShredError hashmap_str_insert(HashMapStr* map, const char* key, Rank value);
  ShredError reverse_map_insert(ReverseMap* map, Rank key, const uint8_t* value, size_t value_len);
}

#endif  //!__HASH__H__
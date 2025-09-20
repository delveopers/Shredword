/**
  @file hashmap.h
  @brief HashMap implementations for storing encoder/decoder mappings in BPE tokenization

  * This file provides hash map data structures and functions for:
  * - Regular hash maps (byte keys -> rank values) for token encoding
  * - String hash maps (string keys -> rank values) for special token encoding  
  * - Reverse maps (rank keys -> byte values) for token decoding
  * Used internally by the CoreBPE tokenizer for fast token lookup and conversion.
*/

#ifndef __HASHMAP__H__
#define __HASHMAP__H__

#include <stddef.h>
#include <stdint.h>

#define DEFAULT_HASH_BUCKET_SIZE 1024
#define DEFAULT_STR_BUCKET_SIZE 256
typedef uint32_t Rank;

typedef struct HashMapNode {
  uint8_t* key;
  size_t key_len;
  Rank value;
  struct HashMapNode* next;
} HashMapNode;

typedef struct HashMap {
  HashMapNode** buckets;
  size_t bucket_count, size;
} HashMap;

typedef struct HashMapStrNode {
  char* key;
  Rank value;
  struct HashMapStrNode* next;
} HashMapStrNode;

typedef struct HashMapStr {
  HashMapStrNode** buckets;
  size_t bucket_count, size;
} HashMapStr;

typedef struct ReverseMapNode {
  Rank key;
  uint8_t* value;
  size_t value_len;
  struct ReverseMapNode* next;
} ReverseMapNode;

typedef struct RvereseMap {
  ReverseMapNode** buckets;
  size_t bucket_count, size;
} ReverseMap;

extern "C" {
  HashMap* hashmapCreate(size_t bucket_count);
  void hashmapFree(HashMap* map);
  bool hashmapGet(HashMap* map, const uint8_t* key, size_t key_len, Rank* value);
  void hashmapInsert(HashMap* map, const uint8_t* key, size_t key_len, Rank value);  

  HashMapStr* strmapCreate(size_t bucket_count);
  void strmapFree(HashMapStr* strmap);
  bool strmapGet(HashMapStr* strmap, const char* key, Rank* value);
  void strmapInsert(HashMapStr* strmap, const char* key, Rank value);

  ReverseMap* revmapCreate(size_t bucket_count);
  void revmapFree(ReverseMap* revmap);
  bool revmapGet(ReverseMap* revmap, Rank key, uint8_t** value, size_t* value_len);
  void revmapInsert(ReverseMap* revmap, Rank key, const uint8_t* value, size_t value_len);
}

#endif  //!__HASHMAP__H__
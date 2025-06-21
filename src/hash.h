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

typedef ShredError ShredError; // forward declaration

extern "C" {
  HashMap* hashmap_new(size_t bucket_count);
  void hashmap_free(HashMap* map);
  bool hashmap_get(HashMap* map, const uint8_t* key, size_t key_len, Rank* value);
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
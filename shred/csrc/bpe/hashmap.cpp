#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "hashmap.h"
#include "core.h"
#include "../inc/hash.h"

// default sizes come from headers, but we choose a power-of-two bucket count for fast masking. We'll expand when load gets high
static inline uint32_t fnv1a_hash_str(const char* str) {
  uint32_t hash = 2166136261u;
  while (*str) {
    hash ^= (uint8_t)*str++;
    hash *= 16777619u;
  }
  return hash;
}

// helper: next power-of-two (returns at least 1)
static inline size_t next_pow2(size_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
#if SIZE_MAX > 0xFFFFFFFF
  v |= v >> 32;
#endif
  v++;
  return v;
}

static void hashmap_rehash(HashMap* map, size_t new_bucket_count) {
  if (!map || new_bucket_count == 0) return;
  new_bucket_count = next_pow2(new_bucket_count);
  HashMapNode** new_buckets = (HashMapNode**)calloc(new_bucket_count, sizeof(HashMapNode*));
  if (!new_buckets) {
    fprintf(stderr, "SHRED>ERROR 102 <hashmap_rehash>: couldn't allocate new buckets\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < map->bucket_count; ++i) {
    HashMapNode* node = map->buckets[i];
    while (node) {
      HashMapNode* next = node->next;
      uint32_t h = fnv1a_hash(node->key, node->key_len);
      size_t bucket = (size_t)h & (new_bucket_count - 1);
      node->next = new_buckets[bucket];
      new_buckets[bucket] = node;
      node = next;
    }
  }
  free(map->buckets);
  map->buckets = new_buckets;
  map->bucket_count = new_bucket_count;
}

static void strmap_rehash(HashMapStr* map, size_t new_bucket_count) {
  if (!map || new_bucket_count == 0) return;
  new_bucket_count = next_pow2(new_bucket_count);
  HashMapStrNode** new_buckets = (HashMapStrNode**)calloc(new_bucket_count, sizeof(HashMapStrNode*));
  if (!new_buckets) {
    fprintf(stderr, "SHRED>ERROR 102 <strmap_rehash>: couldn't allocate new buckets\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < map->bucket_count; ++i) {
    HashMapStrNode* node = map->buckets[i];
    while (node) {
      HashMapStrNode* next = node->next;
      uint32_t h = fnv1a_hash_str(node->key);
      size_t bucket = (size_t)h & (new_bucket_count - 1);
      node->next = new_buckets[bucket];
      new_buckets[bucket] = node;
      node = next;
    }
  }
  free(map->buckets);
  map->buckets = new_buckets;
  map->bucket_count = new_bucket_count;
}

static void revmap_rehash(ReverseMap* map, size_t new_bucket_count) {
  if (!map || new_bucket_count == 0) return;
  new_bucket_count = next_pow2(new_bucket_count);
  ReverseMapNode** new_buckets = (ReverseMapNode**)calloc(new_bucket_count, sizeof(ReverseMapNode*));
  if (!new_buckets) {
    fprintf(stderr, "SHRED>ERROR 102 <revmap_rehash>: couldn't allocate new buckets\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < map->bucket_count; ++i) {
    ReverseMapNode* node = map->buckets[i];
    while (node) {
      ReverseMapNode* next = node->next;
      size_t bucket = (size_t)(node->key) & (new_bucket_count - 1);
      node->next = new_buckets[bucket];
      new_buckets[bucket] = node;
      node = next;
    }
  }
  free(map->buckets);
  map->buckets = new_buckets;
  map->bucket_count = new_bucket_count;
}

HashMap* hashmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = DEFAULT_HASH_BUCKET_SIZE;
  bucket_count = next_pow2(bucket_count);
  HashMap* map = (HashMap*)malloc(sizeof(HashMap));
  if (!map) return NULL;
  map->buckets = (HashMapNode**)calloc(bucket_count, sizeof(HashMapNode*));
  if (!map->buckets) { free(map); return NULL; }
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void hashmapFree(HashMap* map) {
  if (!map) return; // make free safe
  for (size_t i = 0; i < map->bucket_count; i++) {
    HashMapNode* node = map->buckets[i];
    while (node) {
      HashMapNode* next = node->next;
      free(node->key);
      free(node);
      node = next;
    }
  }
  free(map->buckets);
  free(map);
}

bool hashmapGet(HashMap* map, const uint8_t* key, size_t key_len, Rank* value) {
  if (!map || !key || !value) return false;
  uint32_t hash = fnv1a_hash(key, key_len);
  size_t bucket = (size_t)hash & (map->bucket_count - 1);
  HashMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key_len == key_len && memcmp(node->key, key, key_len) == 0) {
      *value = node->value;
      return true;
    }
    node = node->next;
  }
  return false;
}

HashMapStr* strmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = DEFAULT_STR_BUCKET_SIZE;
  bucket_count = next_pow2(bucket_count);
  HashMapStr* strmap = (HashMapStr*)malloc(sizeof(HashMapStr));
  if (!strmap) return NULL;
  strmap->buckets = (HashMapStrNode**)calloc(bucket_count, sizeof(HashMapStrNode*));
  if (!strmap->buckets) { free(strmap); return NULL; }
  strmap->bucket_count = bucket_count;
  strmap->size = 0;
  return strmap;
}

void strmapFree(HashMapStr* map) {
  if (!map) return;
  for (size_t i = 0; i < map->bucket_count; i++) {
    HashMapStrNode* node = map->buckets[i];
    while (node) {
      HashMapStrNode* next = node->next;
      free(node->key);
      free(node);
      node = next;
    }
  }
  free(map->buckets);
  free(map);
}

bool strmapGet(HashMapStr* map, const char* key, Rank* value) {
  if (!map || !key || !value) return false;
  uint32_t hash = fnv1a_hash_str(key);
  size_t bucket = (size_t)hash & (map->bucket_count - 1);
  HashMapStrNode* node = map->buckets[bucket];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      *value = node->value;
      return true;
    }
    node = node->next;
  }
  return false;
}

ReverseMap* revmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = DEFAULT_HASH_BUCKET_SIZE;
  bucket_count = next_pow2(bucket_count);
  ReverseMap* map = (ReverseMap*)malloc(sizeof(ReverseMap));
  if (!map) return NULL;
  map->buckets = (ReverseMapNode**)calloc(bucket_count, sizeof(ReverseMapNode*));
  if (!map->buckets) { free(map); return NULL; }
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void revmapFree(ReverseMap* map) {
  if (!map) return;
  for (size_t i = 0; i < map->bucket_count; i++) {
    ReverseMapNode* node = map->buckets[i];
    while (node) {
      ReverseMapNode* next = node->next;
      free(node->value);
      free(node);
      node = next;
    }
  }
  free(map->buckets);
  free(map);
}

bool revmapGet(ReverseMap* map, Rank key, uint8_t** value, size_t* value_len) {
  if (!map || !value || !value_len) return false;
  size_t bucket = (size_t)key & (map->bucket_count - 1);
  ReverseMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key == key) {
      *value = node->value;
      *value_len = node->value_len;
      return true;
    }
    node = node->next;
  }
  return false;
}

void hashmapInsert(HashMap* map, const uint8_t* key, size_t key_len, Rank value) {
  if (!map || !key) {
    fprintf(stderr, "SHRED>ERROR 101 <hashmapInsert() in hashmap.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (map->size >= map->bucket_count) hashmap_rehash(map, map->bucket_count * 2);

  uint32_t hash = fnv1a_hash(key, key_len);
  size_t bucket = (size_t)hash & (map->bucket_count - 1);
  HashMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key_len == key_len && memcmp(node->key, key, key_len) == 0) {
      node->value = value;
      return;
    }
    node = node->next;
  }

  node = (HashMapNode*)malloc(sizeof(HashMapNode));
  if (!node) {
    fprintf(stderr, "SHRED>ERROR 102 <hashmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }

  // handle zero-length keys safely by allocating at least 1 byte
  size_t alloc_len = (key_len == 0) ? 1 : key_len;
  node->key = (uint8_t*)malloc(alloc_len);
  if (!node->key) {
    free(node);
    fprintf(stderr, "SHRED>ERROR 102 <hashmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  if (key_len > 0) memcpy(node->key, key, key_len);
  node->key_len = key_len;
  node->value = value;
  node->next = map->buckets[bucket];
  map->buckets[bucket] = node;
  map->size++;
}

void strmapInsert(HashMapStr* strmap, const char* key, Rank value) {
  if (!strmap || !key) {
    fprintf(stderr, "SHRED>ERROR 101 <strmapInsert() in hashmap.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

  if (strmap->size >= strmap->bucket_count) strmap_rehash(strmap, strmap->bucket_count * 2);
  uint32_t hash = fnv1a_hash_str(key);
  size_t bucket = (size_t)hash & (strmap->bucket_count - 1);
  HashMapStrNode* node = strmap->buckets[bucket];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      node->value = value;
      return;
    }
    node = node->next;
  }
  node = (HashMapStrNode*)malloc(sizeof(HashMapStrNode));
  if (!node) {
    fprintf(stderr, "SHRED>ERROR 102 <strmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->key = strdup(key);
  if (!node->key) {
    free(node);
    fprintf(stderr, "SHRED>ERROR 102 <strmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->value = value;
  node->next = strmap->buckets[bucket];
  strmap->buckets[bucket] = node;
  strmap->size++;
}

void revmapInsert(ReverseMap* revmap, Rank key, const uint8_t* value, size_t value_len) {
  if (!revmap || !value) {
    fprintf(stderr, "SHRED>ERROR 101 <revmapInsert() in hashmap.c>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  if (revmap->size >= revmap->bucket_count) revmap_rehash(revmap, revmap->bucket_count * 2);

  size_t bucket = (size_t)key & (revmap->bucket_count - 1);
  ReverseMapNode* node = revmap->buckets[bucket];
  while (node) {
    if (node->key == key) {
      free(node->value);
      node->value = (uint8_t*)malloc(value_len ? value_len : 1);
      if (!node->value) {
        fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      if (value_len) memcpy(node->value, value, value_len);
      node->value_len = value_len;
      return;
    }
    node = node->next;
  }
  node = (ReverseMapNode*)malloc(sizeof(ReverseMapNode));
  if (!node) {
    fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->value = (uint8_t*)malloc(value_len ? value_len : 1);
  if (!node->value) {
    free(node);
    fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.c>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  if (value_len) memcpy(node->value, value, value_len);
  node->key = key;
  node->value_len = value_len;
  node->next = revmap->buckets[bucket];
  revmap->buckets[bucket] = node;
  revmap->size++;
}
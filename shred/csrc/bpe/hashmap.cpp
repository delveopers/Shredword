#include <stdlib.h>
#include <string.h>
#include "hashmap.h"

static inline uint32_t hash_bytes(const uint8_t* data, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; i++) {
    hash ^= data[i];
    hash *= 16777619u;
  }
  return hash;
}

static inline uint32_t hash_str(const char* s) {
  uint32_t hash = 2166136261u;
  while (*s) {
    hash ^= (uint8_t)*s++;
    hash *= 16777619u;
  }
  return hash;
}

HashMap* hashmapCreate(size_t bucket_count) {
  HashMap* map = (HashMap*)malloc(sizeof(HashMap));
  map->bucket_count = bucket_count ? bucket_count : DEFAULT_HASH_BUCKET_SIZE;
  map->size = 0;
  map->buckets = (HashMapNode**)calloc(map->bucket_count, sizeof(HashMapNode*));
  return map;
}

void hashmapFree(HashMap* map) {
  if (!map) return;
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
  uint32_t h = hash_bytes(key, key_len);
  size_t idx = h % map->bucket_count;
  HashMapNode* node = map->buckets[idx];
  while (node) {
    if (node->key_len == key_len &&
        memcmp(node->key, key, key_len) == 0) {
      *value = node->value;
      return true;
    }
    node = node->next;
  }
  return false;
}

void hashmapInsert(HashMap* map, const uint8_t* key, size_t key_len, Rank value) {
  uint32_t h = hash_bytes(key, key_len);
  size_t idx = h % map->bucket_count;
  HashMapNode* node = map->buckets[idx];

  while (node) {
    if (node->key_len == key_len &&
        memcmp(node->key, key, key_len) == 0) {
      node->value = value;
      return;
    }
    node = node->next;
  }

  HashMapNode* new_node = (HashMapNode*)malloc(sizeof(HashMapNode));
  new_node->key = (uint8_t*)malloc(key_len);
  memcpy(new_node->key, key, key_len);
  new_node->key_len = key_len;
  new_node->value = value;
  new_node->next = map->buckets[idx];
  map->buckets[idx] = new_node;
  map->size++;
}

HashMapStr* strmapCreate(size_t bucket_count) {
  HashMapStr* map = (HashMapStr*)malloc(sizeof(HashMapStr));
  map->bucket_count = bucket_count ? bucket_count : DEFAULT_STR_BUCKET_SIZE;
  map->size = 0;
  map->buckets = (HashMapStrNode**)calloc(map->bucket_count, sizeof(HashMapStrNode*));
  return map;
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
  uint32_t h = hash_str(key);
  size_t idx = h % map->bucket_count;
  HashMapStrNode* node = map->buckets[idx];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      *value = node->value;
      return true;
    }
    node = node->next;
  }
  return false;
}

void strmapInsert(HashMapStr* map, const char* key, Rank value) {
  uint32_t h = hash_str(key);
  size_t idx = h % map->bucket_count;
  HashMapStrNode* node = map->buckets[idx];

  while (node) {
    if (strcmp(node->key, key) == 0) {
      node->value = value;
      return;
    }
    node = node->next;
  }

  HashMapStrNode* new_node = (HashMapStrNode*)malloc(sizeof(HashMapStrNode));
  size_t len = strlen(key);
  new_node->key = (char*)malloc(len + 1);
  memcpy(new_node->key, key, len + 1);
  new_node->value = value;
  new_node->next = map->buckets[idx];
  map->buckets[idx] = new_node;
  map->size++;
}

ReverseMap* revmapCreate(size_t bucket_count) {
  ReverseMap* map = (ReverseMap*)malloc(sizeof(ReverseMap));
  map->bucket_count = bucket_count ? bucket_count : DEFAULT_HASH_BUCKET_SIZE;
  map->size = 0;
  map->buckets = (ReverseMapNode**)calloc(map->bucket_count, sizeof(ReverseMapNode*));
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
  size_t idx = key % map->bucket_count;
  ReverseMapNode* node = map->buckets[idx];
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

void revmapInsert(ReverseMap* map, Rank key, const uint8_t* value, size_t value_len) {
  size_t idx = key % map->bucket_count;
  ReverseMapNode* node = map->buckets[idx];

  while (node) {
    if (node->key == key) {
      free(node->value);
      node->value = (uint8_t*)malloc(value_len);
      memcpy(node->value, value, value_len);
      node->value_len = value_len;
      return;
    }
    node = node->next;
  }

  ReverseMapNode* new_node = (ReverseMapNode*)malloc(sizeof(ReverseMapNode));
  new_node->key = key;
  new_node->value = (uint8_t*)malloc(value_len);
  memcpy(new_node->value, value, value_len);
  new_node->value_len = value_len;
  new_node->next = map->buckets[idx];
  map->buckets[idx] = new_node;
  map->size++;
}
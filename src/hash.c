#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "hash.h"
#include "core.h"

// FNV-1a hash function
static uint32_t fnv1a_hash(const uint8_t* data, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; i++) {
    hash ^= data[i];
    hash *= 16777619u;
  }
  return hash;
}

static uint32_t fnv1a_hash_str(const char* str) {
  return fnv1a_hash((const uint8_t*)str, strlen(str));
}

// Hash map implementation
HashMap* hashmap_new(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = 1024;
  
  HashMap* map = malloc(sizeof(HashMap));
  if (!map) return NULL;
  
  map->buckets = calloc(bucket_count, sizeof(HashMapNode*));
  if (!map->buckets) {
    free(map);
    return NULL;
  }
  
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void hashmap_free(HashMap* map) {
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

bool hashmap_get(HashMap* map, const uint8_t* key, size_t key_len, Rank* value) {
  if (!map || !key || !value) return false;
  
  uint32_t hash = fnv1a_hash(key, key_len);
  size_t bucket = hash % map->bucket_count;
  
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

// String hash map implementation
HashMapStr* hashmap_str_new(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = 256;
  
  HashMapStr* map = (HashMapStr*)malloc(sizeof(HashMapStr));
  if (!map) return NULL;
  
  map->buckets = calloc(bucket_count, sizeof(HashMapStrNode*));
  if (!map->buckets) {
    free(map);
    return NULL;
  }
  
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void hashmap_str_free(HashMapStr* map) {
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

bool hashmap_str_get(HashMapStr* map, const char* key, Rank* value) {
  if (!map || !key || !value) return false;
  
  uint32_t hash = fnv1a_hash_str(key);
  size_t bucket = hash % map->bucket_count;
  
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

// Reverse map implementation
ReverseMap* reverse_map_new(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = 1024;
  
  ReverseMap* map = malloc(sizeof(ReverseMap));
  if (!map) return NULL;
  
  map->buckets = calloc(bucket_count, sizeof(ReverseMapNode*));
  if (!map->buckets) {
    free(map);
    return NULL;
  }
  
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void reverse_map_free(ReverseMap* map) {
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

bool reverse_map_get(ReverseMap* map, Rank key, uint8_t** value, size_t* value_len) {
  if (!map || !value || !value_len) return false;
  
  size_t bucket = key % map->bucket_count;
  
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

ShredError hashmap_insert(HashMap* map, const uint8_t* key, size_t key_len, Rank value) {
  if (!map || !key) return ERROR_NULL_POINTER;
  
  uint32_t hash = fnv1a_hash(key, key_len);
  size_t bucket = hash % map->bucket_count;
  
  // Check if key already exists
  HashMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key_len == key_len && memcmp(node->key, key, key_len) == 0) {
      node->value = value; // Update existing
      return OK;
    }
    node = node->next;
  }

  // Create new node
  node = (HashMapNode*)malloc(sizeof(HashMapNode));
  if (!node) return ERROR_MEMORY_ALLOCATION;

  node->key = malloc(key_len);
  if (!node->key) {
    free(node);
    return ERROR_MEMORY_ALLOCATION;
  }
  
  memcpy(node->key, key, key_len);
  node->key_len = key_len;
  node->value = value;
  node->next = map->buckets[bucket];
  map->buckets[bucket] = node;
  map->size++;

  return OK;
}

ShredError hashmap_str_insert(HashMapStr* map, const char* key, Rank value) {
  if (!map || !key) return ERROR_NULL_POINTER;
  
  uint32_t hash = fnv1a_hash_str(key);
  size_t bucket = hash % map->bucket_count;
  
  // Check if key already exists
  HashMapStrNode* node = map->buckets[bucket];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      node->value = value; // Update existing
      return OK;
    }
    node = node->next;
  }

  // Create new node
  node = (HashMapStrNode*)malloc(sizeof(HashMapStrNode));
  if (!node) return ERROR_MEMORY_ALLOCATION;
  
  node->key = strdup(key);
  if (!node->key) {
    free(node);
    return ERROR_MEMORY_ALLOCATION;
  }
  
  node->value = value;
  node->next = map->buckets[bucket];
  map->buckets[bucket] = node;
  map->size++;
  
  return OK;
}

ShredError reverse_map_insert(ReverseMap* map, Rank key, const uint8_t* value, size_t value_len) {
  if (!map || !value) return ERROR_NULL_POINTER;
  
  size_t bucket = key % map->bucket_count;
  
  // Check if key already exists
  ReverseMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key == key) {
      free(node->value);
      node->value = malloc(value_len);
      if (!node->value) return ERROR_MEMORY_ALLOCATION;
      memcpy(node->value, value, value_len);
      node->value_len = value_len;
      return OK;
    }
    node = node->next;
  }

  // Create new node
  node = (ReverseMapNode*)malloc(sizeof(ReverseMapNode));
  if (!node) return ERROR_MEMORY_ALLOCATION;
  
  node->value = malloc(value_len);
  if (!node->value) {
    free(node);
    return ERROR_MEMORY_ALLOCATION;
  }
  
  memcpy(node->value, value, value_len);
  node->key = key;
  node->value_len = value_len;
  node->next = map->buckets[bucket];
  map->buckets[bucket] = node;
  map->size++;
  
  return OK;
}

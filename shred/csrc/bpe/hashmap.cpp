#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "hashmap.h"
#include "core.h"
#include "inc/hash.h"

static uint32_t fnv1a_hash_str(const char* str) { return fnv1a_hash((const uint8_t*)str, strlen(str)); }

HashMap* hashmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = DEFAULT_HASH_BUCKET_SIZE;
  HashMap* map = (HashMap*)malloc(sizeof(HashMap));
  if (!map) return NULL;
  map->buckets = (HashMapNode**)calloc(bucket_count, sizeof(HashMapNode*));
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void hashmapFree(HashMap* map) {
  if (!map) {
    fprintf(stderr, "SHRED>ERROR 101 <hashmapFree() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
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
  if (!map || !key || !value) {
    fprintf(stderr, "SHRED>ERROR 101 <hashmapGet() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

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

HashMapStr* strmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count == DEFAULT_STR_BUCKET_SIZE;
  HashMapStr* strmap = (HashMapStr*)malloc(sizeof(HashMapStr));
  strmap->buckets = (HashMapStrNode**)calloc(bucket_count, sizeof(HashMapStrNode*));
  strmap->bucket_count = bucket_count;
  strmap->size = 0;
  return strmap;
}

void strmapFree(HashMapStr* map) {
  if (!map) {
    fprintf(stderr, "SHRED>ERROR 101 <strmapFree() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
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
  if (!map || !key || !value) {
    fprintf(stderr, "SHRED>ERROR 101 <strmapGet() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }

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

ReverseMap* revmapCreate(size_t bucket_count) {
  if (bucket_count == 0) bucket_count = DEFAULT_HASH_BUCKET_SIZE;
  ReverseMap* map = (ReverseMap*)malloc(sizeof(ReverseMap));
  if (!map) return NULL;
  map->buckets = (ReverseMapNode**)calloc(bucket_count, sizeof(ReverseMapNode*));
  map->bucket_count = bucket_count;
  map->size = 0;
  return map;
}

void revmapFree(ReverseMap* map) {
  if (!map) {
    fprintf(stderr, "SHRED>ERROR 101 <revmapFree() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
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

void hashmapInsert(HashMap* map, const uint8_t* key, size_t key_len, Rank value) {
  if (!map || !key) {
    fprintf(stderr, "SHRED>ERROR 101 <hashmapInsert() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  uint32_t hash = fnv1a_hash(key, key_len);
  size_t bucket = hash % map->bucket_count;

  HashMapNode* node = map->buckets[bucket];
  while (node) {
    if (node->key == key && memcmp(node->key, key, key_len) == 0) {
      node->value = value;
      return;
    }
    node = node->next;
  }

  node = (HashMapNode*)malloc(sizeof(HashMapNode));
  if (!node) {
    fprintf(stderr, "SHRED>ERROR 102 <hashmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->key = (uint8_t*)malloc(key_len);
  if (!node->key) {
    fprintf(stderr, "SHRED>ERROR 102 <hashmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  memcpy(node->key, key, key_len);
  node->key_len = key_len;
  node->value = value;
  node->next = map->buckets[bucket];
  map->buckets[bucket] = node;
  map->size++;
}

void strmapInsert(HashMapStr* strmap, const char* key, Rank value) {
  if (!strmap || !key ) {
    fprintf(stderr, "SHRED>ERROR 101 <strmapInsert() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE); 
  }

  uint32_t hash = fnv1a_hash_str(key);
  size_t bucket = hash % strmap->bucket_count;

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
    fprintf(stderr, "SHRED>ERROR 102 <strmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->key = strdup(key);
  if (!node->key) {
    fprintf(stderr, "SHRED>ERROR 102 <strmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->value = value;
  node->next = strmap->buckets[bucket];
  strmap->buckets[bucket] = node;
  strmap->size++;
}

void revmapInsert(ReverseMap* revmap, Rank key, const uint8_t* value, size_t value_len) {
  if (!revmap || !value) {
    fprintf(stderr, "SHRED>ERROR 101 <revmapInsert() in hashmap.cpp>:  Invalid or NULL Parameters\n");
    exit(EXIT_FAILURE);
  }
  size_t bucket = key % revmap->bucket_count;
  ReverseMapNode* node = revmap->buckets[bucket];
  while (node) {
    if (node->key == key) {
      free(node->value);
      node->value = (uint8_t*)malloc(value_len);
      if (!node->value) {
        fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
        exit(EXIT_FAILURE);
      }
      memcpy(node->value, value, value_len);
      node->value_len = value_len;
      return;
    }
    node = node->next;
  }

  node = (ReverseMapNode*)malloc(sizeof(ReverseMapNode));
  if (!node) {
    fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }
  node->value = (uint8_t*)malloc(value_len);
  if (!node->value) {
    fprintf(stderr, "SHRED>ERROR 102 <revmapInsert() in hashmap.cpp>:  Couldn't allocate memory\n");
    exit(EXIT_FAILURE);
  }

  memcpy(node->value, value, value_len);
  node->key = key;
  node->value_len = value_len;
  node->next = revmap->buckets[bucket];
  revmap->buckets[bucket] = node;
  revmap->size++;
}
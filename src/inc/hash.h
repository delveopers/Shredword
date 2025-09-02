#ifndef __HASH_H__
#define __HASH_H__

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint32_t djb2_hash(const char *str) {
  uint32_t hash = 5381;
  int c;
  while ((c = *str++)) hash = ((hash << 5) + hash) + c;
  return hash;
}

static inline uint32_t fnv1a_hash(const char *str) {
  uint32_t hash = 2166136261u;
  while (*str) {
    hash ^= (uint8_t)*str++;
    hash *= 16777619;
  }
  return hash;
}

static inline uint32_t sdbm_hash(const char *str) {
  uint32_t hash = 0;
  int c;
  while ((c = *str++)) hash = c + (hash << 6) + (hash << 16) - hash;
  return hash;
}

static inline uint32_t murmur3_hash(const char *key, int len) {
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  const uint32_t r1 = 15;
  const uint32_t r2 = 13;
  const uint32_t m = 5;
  const uint32_t n = 0xe6546b64;
  
  uint32_t hash = 0;
  const int nblocks = len / 4;
  const uint32_t *blocks = (const uint32_t*)key;
  
  for (int i = 0; i < nblocks; i++) {
    uint32_t k = blocks[i];
    k *= c1;
    k = (k << r1) | (k >> (32 - r1));
    k *= c2;
    hash ^= k;
    hash = ((hash << r2) | (hash >> (32 - r2))) * m + n;
  }
  
  const uint8_t *tail = (const uint8_t*)(key + nblocks * 4);
  uint32_t k1 = 0;
  switch (len & 3) {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
      k1 *= c1;
      k1 = (k1 << r1) | (k1 >> (32 - r1));
      k1 *= c2;
      hash ^= k1;
  }
  
  hash ^= len;
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);
  
  return hash;
}

static inline uint32_t int_hash(int key) {
  key = ((key >> 16) ^ key) * 0x45d9f3b;
  key = ((key >> 16) ^ key) * 0x45d9f3b;
  key = (key >> 16) ^ key;
  return key;
}

static inline uint32_t cache_hash(int key, int capacity) {
  return int_hash(key) % capacity;
}

static inline uint32_t heap_hash(const char *token, int capacity) {
  return djb2_hash(token) % capacity;
}

#ifdef __cplusplus
}
#endif

#endif
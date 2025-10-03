#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "../../src/inc/hash.h"

void test_djb2() {
  assert(djb2_hash("hello") == djb2_hash("hello"));
  assert(djb2_hash("") == 5381);
  assert(djb2_hash("a") != djb2_hash("b"));
}

void test_fnv1a() {
  const char *s = "hello";
  uint32_t h1 = fnv1a_hash((const uint8_t*)s, strlen(s));
  uint32_t h2 = fnv1a_hash((const uint8_t*)s, strlen(s));
  assert(h1 == h2);
  assert(fnv1a_hash((const uint8_t*)"", 0) == 2166136261u);
}

void test_sdbm() {
  assert(sdbm_hash("world") == sdbm_hash("world"));
  assert(sdbm_hash("") == 0);
  assert(sdbm_hash("a") != sdbm_hash("b"));
}

void test_murmur3() {
  const char *s = "hashing";
  int len = strlen(s);
  uint32_t h1 = murmur3_hash(s, len);
  uint32_t h2 = murmur3_hash(s, len);
  assert(h1 == h2);
  assert(murmur3_hash("", 0) == murmur3_hash("", 0));
}

void test_int_hash() {
  int k = 12345;
  uint32_t h1 = int_hash(k);
  uint32_t h2 = int_hash(k);
  assert(h1 == h2);
  assert(int_hash(0) == int_hash(0));
}

void test_cache_hash() {
  int cap = 100;
  int k = 55;
  uint32_t h = cache_hash(k, cap);
  assert(h < (uint32_t)cap);
}

void test_heap_hash() {
  int cap = 50;
  uint32_t h = heap_hash("token", cap);
  assert(h < (uint32_t)cap);
  assert(heap_hash("token", cap) == heap_hash("token", cap));
}

int main() {
  test_djb2();
  test_fnv1a();
  test_sdbm();
  test_murmur3();
  test_int_hash();
  test_cache_hash();
  test_heap_hash();

  printf("All hash tests passed successfully!\n");
  return 0;
}

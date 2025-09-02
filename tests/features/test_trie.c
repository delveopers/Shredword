#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "trie.h"

void test_basic_operations() {
  printf("Testing basic operations...\n");
  
  SubwordTrie* trie = trieCreate();
  assert(trie != NULL);
  assert(trieGetTokenCount(trie) == 0);
  
  assert(trieInsert(trie, "hello", 10));
  assert(trieInsert(trie, "world", 5));
  assert(trieInsert(trie, "hi", 15));
  assert(trieGetTokenCount(trie) == 3);
  
  assert(trieContains(trie, "hello"));
  assert(trieContains(trie, "world"));
  assert(trieContains(trie, "hi"));
  assert(!trieContains(trie, "test"));
  
  assert(trieSearch(trie, "hello") == 10);
  assert(trieSearch(trie, "world") == 5);
  assert(trieSearch(trie, "hi") == 15);
  assert(trieSearch(trie, "missing") == -1);
  
  trieDestroy(trie);
  printf("Basic operations: PASSED\n");
}

void test_update_frequency() {
  printf("Testing frequency updates...\n");
  
  SubwordTrie* trie = trieCreate();
  assert(trieInsert(trie, "token", 100));
  assert(trieSearch(trie, "token") == 100);
  
  assert(trieUpdateFreq(trie, "token", 200));
  assert(trieSearch(trie, "token") == 200);
  
  assert(!trieUpdateFreq(trie, "missing", 50));
  
  trieDestroy(trie);
  printf("Frequency updates: PASSED\n");
}

void test_remove_tokens() {
  printf("Testing token removal...\n");
  
  SubwordTrie* trie = trieCreate();
  assert(trieInsert(trie, "remove", 1));
  assert(trieInsert(trie, "keep", 1));
  assert(trieGetTokenCount(trie) == 2);
  
  assert(trieRemove(trie, "remove"));
  assert(!trieContains(trie, "remove"));
  assert(trieContains(trie, "keep"));
  assert(trieGetTokenCount(trie) == 1);
  
  assert(!trieRemove(trie, "missing"));
  
  trieDestroy(trie);
  printf("Token removal: PASSED\n");
}

void test_get_all_tokens() {
  printf("Testing get all tokens...\n");
  
  SubwordTrie* trie = trieCreate();
  assert(trieInsert(trie, "a", 1));
  assert(trieInsert(trie, "b", 2));
  assert(trieInsert(trie, "c", 3));
  
  char** tokens;
  int* freqs;
  int count;
  trieGetAllTokens(trie, &tokens, &freqs, &count);
  
  assert(count == 3);
  printf("Found %d tokens: ", count);
  for (int i = 0; i < count; i++) {
    printf("%s(%d) ", tokens[i], freqs[i]);
  }
  printf("\n");
  
  for (int i = 0; i < count; i++) {
    free(tokens[i]);
  }
  free(tokens);
  free(freqs);
  
  trieDestroy(trie);
  printf("Get all tokens: PASSED\n");
}

void test_edge_cases() {
  printf("Testing edge cases...\n");
  
  SubwordTrie* trie = trieCreate();
  
  assert(!trieInsert(trie, "", 1));
  assert(!trieInsert(NULL, "test", 1));
  assert(!trieInsert(trie, "test", -1));
  
  char long_token[20] = "verylongtoken123";
  assert(!trieInsert(trie, long_token, 1));
  
  assert(trieSearch(NULL, "test") == -1);
  assert(trieSearch(trie, NULL) == -1);
  
  trieDestroy(trie);
  trieDestroy(NULL);
  
  printf("Edge cases: PASSED\n");
}

void test_subword_tokenization_scenario() {
  printf("Testing subword tokenization scenario...\n");
  
  SubwordTrie* trie = trieCreate();
  
  assert(trieInsert(trie, "un", 50));
  assert(trieInsert(trie, "##able", 30));
  assert(trieInsert(trie, "##ing", 40));
  assert(trieInsert(trie, "work", 25));
  assert(trieInsert(trie, "##ed", 35));
  
  printf("Subword vocabulary loaded: %d tokens\n", trieGetTokenCount(trie));
  
  const char* subwords[] = {"un", "##able", "##ing", "work", "##ed"};
  int expected_freqs[] = {50, 30, 40, 25, 35};
  
  for (int i = 0; i < 5; i++) {
    int freq = trieSearch(trie, subwords[i]);
    assert(freq == expected_freqs[i]);
    printf("Token '%s': frequency %d\n", subwords[i], freq);
  }
  
  trieDestroy(trie);
  printf("Subword tokenization scenario: PASSED\n");
}

int main() {
  printf("Running Trie Tests for Unigram Tokenizer\n");
  printf("========================================\n");
  
  test_basic_operations();
  test_update_frequency();
  test_remove_tokens();
  test_get_all_tokens();
  test_edge_cases();
  test_subword_tokenization_scenario();
  
  printf("\nAll tests passed! Trie is ready for unigram tokenizer.\n");
  return 0;
}
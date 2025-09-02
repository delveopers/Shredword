/**
  @brief trie-based dtype to store possible vocabs
  
  * trie-based functions for creating, inserting & deleting tries & entries
  * word-count & printing functions
*/

#ifndef __TRIE__H__
#define __TRIE__H__

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CHARS 256
#define TRIE_CHILDREN 256
#define MAX_TOKEN_LEN 16

typedef struct TrieNode {
  struct TrieNode* children[TRIE_CHILDREN];
  bool is_token;
  int freq;
} TrieNode;

typedef struct SubwordTrie {
  TrieNode* root;
  int total_tokens;
} SubwordTrie;

static TrieNode* trieNodeCreate() {
  TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
  if (!node) return NULL;
  node->is_token = false;
  node->freq = 0;
  return node;
}

static void trieNodeDestroy(TrieNode* node) {
  if (!node) return;
  for (int i = 0; i < TRIE_CHILDREN; i++) {
    if (node->children[i]) trieNodeDestroy(node->children[i]);
  }
  free(node);
}

SubwordTrie* trieCreate() {
  SubwordTrie* trie = (SubwordTrie*)malloc(sizeof(SubwordTrie));
  if (!trie) return NULL;
  trie->root = trieNodeCreate();
  trie->total_tokens = 0;
  return trie;
}

void trieDestroy(SubwordTrie* trie) {
  if (!trie) return;
  trieNodeDestroy(trie->root);
  free(trie);
}

bool trieInsert(SubwordTrie* trie, const char* token, int freq) {
  if (!trie || !token || freq < 0 || strlen(token) == 0 || strlen(token) >= MAX_TOKEN_LEN) return false;
  TrieNode* node = trie->root;
  for (const char* p = token; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (!node->children[c]) {
      node->children[c] = trieNodeCreate();
      if (!node->children[c]) return false;
    }
    node = node->children[c];
  }

  if (!node->is_token) trie->total_tokens++;
  node->is_token = true;
  node->freq = freq;
  return true;
}

int trieSearch(SubwordTrie* trie, const char* token) {
  if (!trie || !token) return -1;

  TrieNode* node = trie->root;
  for (const char* p = token; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (!node->children[c]) return -1;
    node = node->children[c];
  }
  return node->is_token ? node->freq : -1;
}

static bool trieNodeHasChildren(TrieNode* node) {
  if (!node) return false;
  for (int i = 0; i < TRIE_CHILDREN; i++) { if (node->children[i]) return true; }
  return false;
}

static bool trieRemoveHelper(TrieNode* node, const char* token, int depth) {
  if (!node) return false;
  if (token[depth] == '\0') {
    if (!node->is_token) return false;
    node->is_token = false;
    node->freq = 0;
    return !trieNodeHasChildren(node);
  }
  unsigned char c = (unsigned char)token[depth];
  TrieNode* child = node->children[c];
  if (!child) return false;
  bool should_delete_child = trieRemoveHelper(child, token, depth + 1);
  if (should_delete_child) {
    trieNodeDestroy(child);
    node->children[c] = NULL;
  }
  return !node->is_token && !trieNodeHasChildren(node);
}

bool trieContains(SubwordTrie *trie, const char *token) {
  return trieSearch(trie, token) != -1;
}

int trieGetTokenCount(SubwordTrie *trie) {
  return trie ? trie->total_tokens : 0;
}

bool trieRemove(SubwordTrie* trie, const char* token) {
  if (!trie || !token) return false;
  if (!trieContains(trie, token)) return false;
  trieRemoveHelper(trie->root, token, 0);
  trie->total_tokens--;
  return true;
}

bool trieUpdateFreq(SubwordTrie* trie, const char* token, int new_freq) {
  if (!trie || !token || new_freq < 0) return false;

  TrieNode* node = trie->root;
  for (const char* p = token; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (!node->children[c]) return false;
    node = node->children[c];
  }
  if (!node->is_token) return false;
  node->freq = new_freq;
  return true;
}

static void trieCollectTokens(TrieNode* node, char* prefix, int depth, char*** tokens, int** freq, int* count, int* capacity) {
  if (!node || depth >= MAX_TOKEN_LEN - 1) return;

  if (node->is_token) {
    if (*count >= *capacity) {
      int new_capacity = (*capacity) * 2;
      char** new_tokens = (char**)realloc(*tokens, sizeof(char*) * new_capacity);
      int* new_freq = (int*)realloc(*freq, sizeof(int) * new_capacity);
      *tokens = new_tokens, *freq = new_freq, *capacity = new_capacity;
    }

    prefix[depth] = '\0';
    (*tokens)[*count] = strdup(prefix);
    if (!(*tokens)[*count]) return;
    (*freq)[*count] = node->freq;
    (*count)++;
  }
  for (int i = 0; i < TRIE_CHILDREN; i++) {
    if (node->children[i]) {
      prefix[depth] = (char)i;
      trieCollectTokens(node->children[i], prefix, depth + 1, tokens, freq, count, capacity);
    }
  }
}

void trieGetAllTokens(SubwordTrie* trie, char*** tokens, int** freq, int* count) {
  if (!trie || !freq || !count || !tokens) return;
  *count = 0;
  int capacity = 100;
  *tokens = (char**)malloc(capacity * sizeof(char*));
  *freq = (int*)malloc(capacity * sizeof(int));
  char prefix[MAX_TOKEN_LEN];
  memset(prefix, 0, MAX_TOKEN_LEN);
  trieCollectTokens(trie->root, prefix, 0, tokens, freq, count, &capacity);
}

static void trieFreeTokens(char **tokens, int count) {
  if (!tokens) return;
  for (int i = 0; i < count; i++) { free(tokens[i]); }
  free(tokens);
}

#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //!__TRIE__H__
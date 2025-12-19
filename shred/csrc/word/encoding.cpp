#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../inc/hash.h"
#include "encoding.h"

static char* my_strdup_c(const char* s) {
  if (!s) return NULL;
  size_t l = strlen(s);
  char* p = (char*)malloc(l + 1);
  if (!p) return NULL;
  memcpy(p, s, l + 1);
  return p;
}

static int next_power_of_two(int v) {
  if (v < 1) return 1;
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}

static void ht_init(Vocab* v, int cap) {
  int c = next_power_of_two(cap);
  v->ht_cap = c;
  v->ht_keys = (char**)calloc(c, sizeof(char*));
  v->ht_vals = (int*)malloc(sizeof(int)*  c);
  for (int i = 0; i < c; ++i) v->ht_vals[i] = -1;
  v->ht_size = 0;
}

static void ht_free(Vocab* v) {
  if (v->ht_keys) free(v->ht_keys);
  if (v->ht_vals) free(v->ht_vals);
  v->ht_keys = NULL;
  v->ht_vals = NULL;
  v->ht_cap = 0;
  v->ht_size = 0;
}

static void rehash(Vocab* v, int newcap) {
  char* *old_keys = v->ht_keys;
  int* old_vals = v->ht_vals;
  int old_cap = v->ht_cap;
  ht_init(v, newcap);
  for (int i = 0; i < old_cap; ++i) {
    char* k = old_keys[i];
    if (k) {
      unsigned int h = djb2_hash(k);
      int idx = (int)(h & (v->ht_cap - 1));
      while (v->ht_keys[idx]) idx = (idx + 1) & (v->ht_cap - 1);
      v->ht_keys[idx] = old_keys[i];
      v->ht_vals[idx] = old_vals[i];
      v->ht_size++;
    }
  }
  free(old_keys);
  free(old_vals);
}

Vocab* vocab_create(void) {
  Vocab* v = (Vocab*)malloc(sizeof(Vocab));
  v->words = (char**)malloc(sizeof(char*)*  16);
  v->counts = (int*)malloc(sizeof(int)*  16);
  v->df = (int*)malloc(sizeof(int)*  16);
  v->size = 0;
  v->cap = 16;
  v->docs = 0;
  v->ht_keys = NULL;
  v->ht_vals = NULL;
  v->ht_cap = 0;
  v->ht_size = 0;
  ht_init(v, 32);
  return v;
}

void vocab_free(Vocab* v) {
  if (!v) return;
  for (int i = 0; i < v->size; ++i) free(v->words[i]);
  free(v->words);
  free(v->counts);
  free(v->df);
  ht_free(v);
  free(v);
}

static void vocab_grow_if_needed(Vocab* v) {
  if (v->size + 1 >= v->cap) {
    int ncap = v->cap*  2;
    v->words = (char**)realloc(v->words, sizeof(char*)*  ncap);
    v->counts = (int*)realloc(v->counts, sizeof(int)*  ncap);
    v->df = (int*)realloc(v->df, sizeof(int)*  ncap);
    v->cap = ncap;
  }
  if ((double)v->ht_size / (double)v->ht_cap > 0.6) {
    rehash(v, v->ht_cap*  2);
  }
}

int vocab_lookup(Vocab* v, const char* token) {
  if (!v || !token) return -1;
  unsigned long h = djb2_hash(token);
  int idx = (int)(h & (v->ht_cap - 1));
  int start = idx;
  while (v->ht_keys[idx]) {
    if (strcmp(v->ht_keys[idx], token) == 0) return v->ht_vals[idx];
    idx = (idx + 1) & (v->ht_cap - 1);
    if (idx == start) break;
  }
  return -1;
}

int vocab_add(Vocab* v, const char* token) {
  if (!v || !token) return -1;
  int found = vocab_lookup(v, token);
  if (found >= 0) {
    v->counts[found] += 1;
    return found;
  }
  vocab_grow_if_needed(v);
  char* s = my_strdup_c(token);
  int id = v->size++;
  v->words[id] = s;
  v->counts[id] = 1;
  v->df[id] = 0;
  unsigned long h = djb2_hash(token);
  int idx = (int)(h & (v->ht_cap - 1));
  while (v->ht_keys[idx]) idx = (idx + 1) & (v->ht_cap - 1);
  v->ht_keys[idx] = v->words[id];
  v->ht_vals[idx] = id;
  v->ht_size++;
  return id;
}

void vocab_add_document(Vocab* v, const char** tokens, int ntokens) {
  if (!v || !tokens || ntokens <= 0) return;
  v->docs += 1;
  int cap_local = ntokens;
  char** seen = (char**)malloc(sizeof(char*)*  cap_local);
  int seen_n = 0;
  for (int i = 0; i < ntokens; ++i) {
    const char* t = tokens[i];
    int id = vocab_lookup(v, t);
    if (id >= 0) { v->counts[id] += 1; } else { id = vocab_add(v, t); }
    int already = 0;
    for (int j = 0; j < seen_n; ++j) if (strcmp(seen[j], t) == 0) { already = 1; break; }
    if (!already) {
      seen[seen_n++] = (char*)t;
      v->df[id] += 1;
    }
  }
  free(seen);
}

int* encode_ids(Vocab* v, const char** tokens, int ntokens, int* out_n) {
  if (!v || !tokens || ntokens <= 0) { if (out_n)* out_n = 0; return NULL; }
  int* ids = (int*)malloc(sizeof(int)*  ntokens);
  for (int i = 0; i < ntokens; ++i) ids[i] = vocab_lookup(v, tokens[i]);
  if (out_n)* out_n = ntokens;
  return ids;
}

float* encode_tfidf_dense(Vocab* v, const char** tokens, int ntokens) {
  if (!v || !tokens || ntokens <= 0) return NULL;
  int V = v->size;
  float* vec = (float*)calloc(V, sizeof(float));
  int* counts = (int*)calloc(V, sizeof(int));
  for (int i = 0; i < ntokens; ++i) {
    int id = vocab_lookup(v, tokens[i]);
    if (id >= 0) counts[id] += 1;
  }
  for (int i = 0; i < V; ++i) {
    if (counts[i] == 0) continue;
    float tf = (float)counts[i] / (float)ntokens;
    int df = v->df[i];
    float idf = logf((1.0f + (float)v->docs) / (1.0f + (float)df)) + 1.0f;
    vec[i] = tf * idf;
  }
  free(counts);
  return vec;
}

void encode_tfidf_sparse(Vocab* v, const char** tokens, int ntokens, int** out_indices, float** out_values, int* out_n) {
  if (!v || !tokens || ntokens <= 0) { if (out_n)* out_n = 0; if (out_indices)* out_indices = NULL; if (out_values)* out_values = NULL; return; }
  int V = v->size;
  int* counts = (int*)calloc(V, sizeof(int));
  for (int i = 0; i < ntokens; ++i) {
    int id = vocab_lookup(v, tokens[i]);
    if (id >= 0) counts[id] += 1;
  }
  int nz = 0;
  for (int i = 0; i < V; ++i) if (counts[i] > 0) nz++;
  int* idxs = (int*)malloc(sizeof(int)*  nz);
  float* vals = (float*)malloc(sizeof(float)*  nz);
  int p = 0;
  for (int i = 0; i < V; ++i) {
    if (counts[i] == 0) continue;
    float tf = (float)counts[i] / (float)ntokens;
    int df = v->df[i];
    float idf = logf((1.0f + (float)v->docs) / (1.0f + (float)df)) + 1.0f;
    idxs[p] = i;
    vals[p] = tf * idf;
    p++;
  }
  free(counts);
  if (out_indices)* out_indices = idxs; else free(idxs);
  if (out_values)* out_values = vals; else free(vals);
  if (out_n)* out_n = p;
}

char* vocab_save(Vocab* v, const char* path) {
  if (!v || !path) return my_strdup_c("invalid arguments");
  FILE* f = fopen(path, "wb");
  if (!f) return my_strdup_c("unable to open file for writing");
  fprintf(f, "DOCS\t%d\n", v->docs);
  for (int i = 0; i < v->size; ++i) {
    fprintf(f, "%s\t%d\t%d\n", v->words[i], v->counts[i], v->df[i]);
  }
  fclose(f);
  return NULL;
}

char* vocab_load(Vocab** out_v, const char* path) {
  if (!out_v || !path) return my_strdup_c("invalid arguments");
  FILE* f = fopen(path, "rb");
  if (!f) return my_strdup_c("unable to open file for reading");
  Vocab* v = vocab_create();
  char buf[4096];
  while (fgets(buf, sizeof(buf), f)) {
    size_t L = strlen(buf);
    while (L > 0 && (buf[L-1] == '\n' || buf[L-1] == '\r')) { buf[L-1] = '\0'; L--; }
    if (L == 0) continue;
    if (strncmp(buf, "DOCS\t", 5) == 0) {
      int docs = atoi(buf + 5);
      v->docs = docs;
      continue;
    }
    char* tab1 = strchr(buf, '\t');
    if (!tab1) continue;
    char* tab2 = strchr(tab1 + 1, '\t');
    if (!tab2) continue;
    *tab1 = '\0';
    *tab2 = '\0';
    const char* word = buf;
    int cnt = atoi(tab1 + 1);
    int df = atoi(tab2 + 1);
    int id = vocab_add(v, word);
    v->counts[id] = cnt;
    v->df[id] = df;
  }
  fclose(f);
  *out_v = v;
  return NULL;
}

void free_buffer(void* p) {
  if (p) free(p);
}
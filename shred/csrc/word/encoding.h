#ifndef __ENCODING_H__
#define __ENCODING_H__

#include <stddef.h>

typedef struct {
  char **words, **ht_keys;
  int *counts, *df, *ht_vals;
  int size, cap, docs, ht_cap, ht_size;
} Vocab;

extern "C" {
  Vocab* vocab_create(void);
  void vocab_free(Vocab* v);
  int vocab_lookup(Vocab* v, const char* token);
  int vocab_add(Vocab* v, const char* token);
  void vocab_add_document(Vocab* v, const char** tokens, int ntokens);
  int* encode_ids(Vocab* v, const char** tokens, int ntokens, int* out_n);
  float* encode_tfidf_dense(Vocab* v, const char** tokens, int ntokens);
  void encode_tfidf_sparse(Vocab* v, const char** tokens, int ntokens, int** out_indices, float** out_values, int* out_n);
  char* vocab_save(Vocab* v, const char* path);
  char* vocab_load(Vocab** out_v, const char* path);
  void free_buffer(void* p);
}

#endif
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include "../../shred/csrc/word/tokenize.h"
#include "../../shred/csrc/word/encoding.h"

int main() {
  std::vector<std::string> docs = {
    "Hello, world! This is a test.",
    "I can't believe it's already 3.14... isn't it?",
    "He's going to the state-of-the-art lab.",
    "\"Quote\" (parentheses) and ellipses...",
    "U.S.A. vs U.K. Mr. Smith's book.",
    "Don't stop believin'!",
    "She'd've said 'no' if she meant it."
  };

  Vocab* v = vocab_create();
  if (!v) {
    std::cerr << "vocab_create failed\n";
    return 1;
  }

  for (size_t di = 0; di < docs.size(); ++di) {
    const std::string &s = docs[di];
    Tokens t = tokenize(s.c_str());
    std::cout << "DOC " << di << " INPUT: " << s << "\n";
    std::cout << "TOKENS: [";
    for (int i = 0; i < t.count; ++i) {
      std::cout << "'" << t.items[i] << "'" << (i+1 < t.count ? ", " : "");
    }
    std::cout << "]\n";
    std::vector<const char*> ptrs;
    ptrs.reserve(t.count);
    for (int i = 0; i < t.count; ++i) ptrs.push_back(t.items[i]);
    vocab_add_document(v, ptrs.data(), (int)ptrs.size());
    free_tokens(&t);
  }

  std::cout << "\nVOCAB SIZE: " << v->size << " DOCS: " << v->docs << "\n";

  const char* sample = "I can't believe it's already 3.14...";
  Tokens t2 = tokenize(sample);
  std::vector<const char*> sample_ptrs;
  for (int i = 0; i < t2.count; ++i) sample_ptrs.push_back(t2.items[i]);

  int ids_n = 0;
  int* ids = encode_ids(v, sample_ptrs.data(), (int)sample_ptrs.size(), &ids_n);
  std::cout << "\nENCODE IDS (" << ids_n << "): [";
  for (int i = 0; i < ids_n; ++i) {
    if (ids[i] >= 0) std::cout << ids[i];
    else std::cout << "NA";
    std::cout << (i+1 < ids_n ? ", " : "");
  }
  std::cout << "]\n";
  free_buffer(ids);

  float* dense = encode_tfidf_dense(v, sample_ptrs.data(), (int)sample_ptrs.size());
  if (dense) {
    std::cout << "\nDENSE TF-IDF NONZERO (" << v->size << " dims):\n";
    for (int i = 0; i < v->size; ++i) {
      if (dense[i] != 0.0f) {
        std::cout << " id=" << i << " word='" << v->words[i] << "' val=" << dense[i] << "\n";
      }
    }
    free_buffer(dense);
  } else {
    std::cout << "dense tfidf returned null\n";
  }

  int *s_idx = NULL;
  float *s_val = NULL;
  int s_n = 0;
  encode_tfidf_sparse(v, sample_ptrs.data(), (int)sample_ptrs.size(), &s_idx, &s_val, &s_n);
  std::cout << "\nSPARSE TF-IDF (" << s_n << "):\n[";
  for (int i = 0; i < s_n; ++i) {
    std::cout << "(" << s_idx[i] << ", " << s_val[i] << ")" << (i+1 < s_n ? ", " : "");
  }
  std::cout << "]\n";
  free_buffer(s_idx);
  free_buffer(s_val);

  free_tokens(&t2);

  char* save_err = vocab_save(v, "vocab_dump.txt");
  if (save_err) {
    std::cerr << "vocab_save error: " << save_err << "\n";
    free_buffer(save_err);
  } else {
    std::cout << "\nvocab saved to vocab_dump.txt\n";
  }

  Vocab* v2 = NULL;
  char* load_err = vocab_load(&v2, "vocab_dump.txt");
  if (load_err) {
    std::cerr << "vocab_load error: " << load_err << "\n";
    free_buffer(load_err);
  } else {
    std::cout << "loaded vocab size: " << v2->size << " docs: " << v2->docs << "\n";
  }

  vocab_free(v);
  if (v2) vocab_free(v2);

  return 0;
}
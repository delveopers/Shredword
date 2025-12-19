#ifndef __TOKENIZE_H__
#define __TOKENIZE_H__

#include <stddef.h>

typedef struct {
  char **items;
  int count;
  int cap;
} Tokens;

extern "C" {
  Tokens tokenize(const char *text);
  void free_tokens(Tokens *t);
}

#endif
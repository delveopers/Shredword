#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "tokenize.h"

static void tokens_init(Tokens* t) {
  t->count = 0;
  t->cap = 16;
  t->items = (char**)malloc(sizeof(char*)*  t->cap);
}

static void tokens_push_raw(Tokens* t, const char* s, int len) {
  if (len <= 0) return;
  if (t->count >= t->cap) {
    t->cap *= 2;
    t->items = (char**)realloc(t->items, sizeof(char*)*  t->cap);
  }
  char* p = (char*)malloc(len + 1);
  memcpy(p, s, len);
  p[len] = '\0';
  t->items[t->count++] = p;
}

static int ends_with(const char* s, int len, const char* suf) {
  int l = strlen(suf);
  if (l > len) return 0;
  return memcmp(s + len - l, suf, l) == 0;
}

static void tokens_push_with_contraction(Tokens* t, const char* s, int len) {
  if (len <= 0) return;
  const char* contractions[] = {"n't", "'ll", "'re", "'ve", "'s", "'m", "'d"};
  int ncon = sizeof(contractions)/sizeof(contractions[0]);
  for (int i = 0; i < ncon; ++i) {
    const char* suf = contractions[i];
    int sl = strlen(suf);
    if (len > sl && ends_with(s, len, suf)) {
      tokens_push_raw(t, s, len - sl);
      tokens_push_raw(t, s + len - sl, sl);
      return;
    }
  }
  tokens_push_raw(t, s, len);
}

static int is_connecting_hyphen(char prev, char next) {
  return (isalnum((unsigned char)prev) && isalnum((unsigned char)next));
}

Tokens tokenize(const char* text) {
  Tokens out;
  tokens_init(&out);
  int n = strlen(text), blen = 0;
  char* buf = (char*)malloc(n + 1);
  for (int i = 0; i < n; ++i) {
    unsigned char c = text[i];
    if (c == '.' ) {
      int j = i, dots = 0;
      while (j < n && text[j] == '.') { dots++; j++; }
      if (dots >= 2) {
        if (blen) {
          tokens_push_with_contraction(&out, buf, blen);
          blen = 0;
        }
        tokens_push_raw(&out, "...", dots);
        i = j - 1;
        continue;
      }
      int prev_digit = (blen > 0 && isdigit((unsigned char)buf[blen-1])), next_digit = (j < n && isdigit((unsigned char)text[j]));
      if (prev_digit && next_digit) {
        buf[blen++] = '.';
        continue;
      } else {
        if (blen) {
          tokens_push_with_contraction(&out, buf, blen);
          blen = 0;
        }
        char tmp = '.';
        tokens_push_raw(&out, &tmp, 1);
        continue;
      }
    }
    if (c == '\'' ) {
      if (blen == 0) {
        char tmp = '\'';
        tokens_push_raw(&out, &tmp, 1);
        continue;
      } else {
        buf[blen++] = '\'';
        continue;
      }
    }
    if (c == '-' ) {
      char prev = (blen>0) ? buf[blen-1] : 0;
      char next = (i+1<n) ? text[i+1] : 0;
      if (is_connecting_hyphen(prev, next)) {
        buf[blen++] = '-';
        continue;
      } else {
        if (blen) {
          tokens_push_with_contraction(&out, buf, blen);
          blen = 0;
        }
        char tmp = '-';
        tokens_push_raw(&out, &tmp, 1);
        continue;
      }
    }
    if (isalnum(c)) {
      buf[blen++] = c;
      continue;
    }
    if (isspace(c)) {
      if (blen) {
        tokens_push_with_contraction(&out, buf, blen);
        blen = 0;
      }
      continue;
    }
    if (c == '"' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}' || c == ',' || c == ':' || c == ';' || c == '?' || c == '!' || c == '/' || c == '\\' ||
        c == '`' || c == '*' || c == '&' || c == '%' || c == '$' || c == '@' || c == '^' || c == '~' || c == '<' || c == '>' || c == '=') {
      if (blen) {
        tokens_push_with_contraction(&out, buf, blen);
        blen = 0;
      }
      char tmp = c;
      tokens_push_raw(&out, &tmp, 1);
      continue;
    }
    buf[blen++] = c;
  }
  if (blen) tokens_push_with_contraction(&out, buf, blen);
  free(buf);
  return out;
}

void free_tokens(Tokens* t) {
  for (int i = 0; i < t->count; ++i) free(t->items[i]);
  free(t->items);
  t->items = NULL;
  t->count = 0;
  t->cap = 0;
}
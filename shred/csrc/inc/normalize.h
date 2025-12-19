#ifndef __NORMALIZE_H__
#define __NORMALIZE_H__

#include <stddef.h>
#include <stdint.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_LINE 16384
#define SPACE_MARKER "\xE2\x96\x81"
#define SPACE_MARKER_LEN 3
#define BATCH_SIZE 64

typedef struct {
  char* data;
  size_t length, capacity;
} NormalizedText;

static const uint32_t SPACE_MARKER_U32 = 0x8196E2;
// replaced function below for C++ support. had to declare whitespaces explicitly, C++ is dumb
// static const unsigned char whitespace_table[256] = {[' '] = 1, ['\t'] = 1, ['\n'] = 1, ['\r'] = 1, ['\v'] = 1, ['\f'] = 1};
static const unsigned char whitespace_table[256] = {
  0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

static inline int is_whitespace(unsigned char c) {
  return whitespace_table[c];
}

static inline int is_space_marker(const char* p) {
  return ((uint32_t*)p)[0] == SPACE_MARKER_U32;
}

static inline NormalizedText* create_normalized_text(size_t capacity) {
  if (capacity == 0) capacity = MAX_LINE;
  NormalizedText* nt = (NormalizedText*)malloc(sizeof(NormalizedText));
  if (!nt) return NULL;

  nt->data = (char*)malloc(capacity);
  if (!nt->data) {
    free(nt);
    return NULL;
  }

  nt->capacity = capacity;
  nt->length = 0;
  nt->data[0] = '\0';
  return nt;
}

static inline void free_normalized_text(NormalizedText* nt) {
  if (nt) {
    if (nt->data) free(nt->data);
    free(nt);
  }
}

static inline int resize_normalized_text(NormalizedText* nt, size_t new_capacity) {
  if (!nt || new_capacity <= nt->capacity) return 0;

  char* new_data = (char*)realloc(nt->data, new_capacity);
  if (!new_data) return -1;

  nt->data = new_data;
  nt->capacity = new_capacity;
  return 0;
}

static inline int append_char(NormalizedText* nt, char c) {
  if (nt->length + 2 >= nt->capacity) {
    if (resize_normalized_text(nt, nt->capacity * 2) != 0) return -1;
  }
  nt->data[nt->length++] = c;
  nt->data[nt->length] = '\0';
  return 0;
}

static inline int append_space_marker(NormalizedText* nt) {
  if (nt->length + SPACE_MARKER_LEN + 1 >= nt->capacity) {
    if (resize_normalized_text(nt, nt->capacity * 2) != 0) return -1;
  }
  memcpy(nt->data + nt->length, SPACE_MARKER, SPACE_MARKER_LEN);
  nt->length += SPACE_MARKER_LEN;
  nt->data[nt->length] = '\0';
  return 0;
}

static inline int normalize_text_fast(const char* input, NormalizedText* output) {
  if (!input || !output) return -1;  
  const char* in = input;
  size_t input_len = strlen(input);
  int in_whitespace = 1, prev_was_space = 1;

  output->length = 0;
  if (output->capacity <= input_len * 2) {
    if (resize_normalized_text(output, input_len * 2 + 256) != 0) return -1;
  }

  for (size_t i = 0; i < input_len; i++) {
    unsigned char c = input[i];

    if (is_whitespace(c)) {
      if (!prev_was_space) {
        if (append_space_marker(output) != 0) return -1;
        prev_was_space = 1;
      }
    } else {
      char lower_c = tolower(c);
      if (append_char(output, lower_c) != 0) return -1;
      prev_was_space = 0;
    }
  }  
  if (output->length >= SPACE_MARKER_LEN && is_space_marker(output->data + output->length - SPACE_MARKER_LEN)) {
    output->length -= SPACE_MARKER_LEN;
    output->data[output->length] = '\0';
  }  
  return 0;
}

static inline int normalize_line_simple(const char* input, char* output, size_t output_size) {
  if (!input || !output || output_size == 0) return -1;

  const char* in = input;
  char* out = output;
  char* out_end = output + output_size - 1;
  int prev_was_space = 1;
  while (*in && out < out_end - SPACE_MARKER_LEN) {
    if (is_whitespace(*in)) {
      if (!prev_was_space) {
        if (out + SPACE_MARKER_LEN > out_end) break;
        memcpy(out, SPACE_MARKER, SPACE_MARKER_LEN);
        out += SPACE_MARKER_LEN;
        prev_was_space = 1;
      }
    } else {
      *out++ = tolower(*in);
      prev_was_space = 0;
    }
    in++;
  }

  if (out >= output + SPACE_MARKER_LEN && is_space_marker(out - SPACE_MARKER_LEN)) out -= SPACE_MARKER_LEN;
  *out = '\0';
  return out - output;
}

static inline int normalize_batch(char** inputs, size_t count, NormalizedText** outputs) {
  if (!inputs || !outputs || count == 0) return -1;

  for (size_t i = 0; i < count; i++) {
    if (!outputs[i]) {
      outputs[i] = create_normalized_text(0);
      if (!outputs[i]) return -1;
    }
    if (normalize_text_fast(inputs[i], outputs[i]) != 0) return -1;
  }
  return 0;
}

static inline int normalize_file(const char* input_path, const char* output_path) {
  FILE *in_file = fopen(input_path, "r"), *out_file = fopen(output_path, "w");
  if (!in_file || !out_file) {
    if (in_file) fclose(in_file);
    if (out_file) fclose(out_file);
    return -1;
  }

  char line_buffer[MAX_LINE];
  NormalizedText* nt = create_normalized_text(MAX_LINE * 2);
  if (!nt) {
    fclose(in_file);
    fclose(out_file);
    return -1;
  }

  size_t line_count = 0;
  while (fgets(line_buffer, MAX_LINE, in_file)) {
    size_t len = strlen(line_buffer);
    if (len > 0 && line_buffer[len - 1] == '\n') {
      line_buffer[len - 1] = '\0';
      len--;
    }

    if (normalize_text_fast(line_buffer, nt) == 0) {
      fprintf(out_file, "%s\n", nt->data);
      line_count++;
    }
  }

  free_normalized_text(nt);
  fclose(in_file);
  fclose(out_file);
  return line_count;
}

static inline void print_normalized_stats(const NormalizedText* nt) {
  if (!nt) return;

  size_t space_markers = 0, chars = 0;
  for (size_t i = 0; i < nt->length; i += 3) {
    if (is_space_marker(nt->data + i)) {
      space_markers++;
      i += 2;
    } else {
      chars++;
      i -= 2;
    }
  }

  printf("Length: %zu, Space markers: %zu, Characters: %zu\n", nt->length, space_markers, chars);
}

static inline const char* get_normalized_data(const NormalizedText* nt) {
  return nt ? nt->data : NULL;
}

static inline size_t get_normalized_length(const NormalizedText* nt) {
  return nt ? nt->length : 0;
}

#ifdef __cplusplus
}
#endif

#endif
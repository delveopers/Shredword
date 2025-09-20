#include "../../src/inc/normalize.h"
#include <time.h>

void test_basic_normalization() {
  printf("=== Basic Normalization Tests ===\n");

  const char* test_cases[] = {
    "Hello World",
    "  Multiple   Spaces  ",
    "MixedCASE text",
    "Tab\tand\nNewline",
    "Numbers123 and SYMBOLS!@#",
    "",
    "   ",
    "SingleWord"
  };

  size_t num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
  NormalizedText* nt = create_normalized_text(1024);

  for (size_t i = 0; i < num_tests; i++) {
    if (normalize_text_fast(test_cases[i], nt) == 0) {
      printf("Input:  '%s'\n", test_cases[i]);
      printf("Output: '%s'\n", nt->data);
      print_normalized_stats(nt);
      printf("\n");
    }
  }
  
  free_normalized_text(nt);
}

void test_line_normalization() {
  printf("=== Line Normalization Test ===\n");
  
  char output_buffer[MAX_LINE];
  const char* input = "This is a TEST line with    MULTIPLE spaces";
  
  int result = normalize_line_simple(input, output_buffer, MAX_LINE);
  if (result >= 0) {
    printf("Input:  '%s'\n", input);
    printf("Output: '%s'\n", output_buffer);
    printf("Length: %d\n\n", result);
  }
}

void test_batch_normalization() {
  printf("=== Batch Normalization Test ===\n");
  
  char* batch_inputs[] = {
    "First line with CAPS",
    "Second   line   with   spaces",
    "Third line\twith\ttabs"
  };
  
  size_t batch_size = sizeof(batch_inputs) / sizeof(batch_inputs[0]);
  NormalizedText** outputs = (NormalizedText**)malloc(batch_size * sizeof(NormalizedText*));
  
  for (size_t i = 0; i < batch_size; i++) {
    outputs[i] = NULL;
  }
  
  if (normalize_batch(batch_inputs, batch_size, outputs) == 0) {
    for (size_t i = 0; i < batch_size; i++) {
      printf("Batch %zu: '%s' -> '%s'\n", i + 1, batch_inputs[i], outputs[i]->data);
    }
  }

  for (size_t i = 0; i < batch_size; i++) {
    free_normalized_text(outputs[i]);
  }
  free(outputs);
  printf("\n");
}

void create_sample_input_file() {
  FILE* f = fopen("new.txt", "w");
  if (!f) return;

  fprintf(f, "This is a SAMPLE text file.\n");
  fprintf(f, "It contains   MULTIPLE    spaces and CAPS.\n");
  fprintf(f, "Some lines have\ttabs and\nnewlines.\n");
  fprintf(f, "Numbers like 123 and symbols like @#$ are included.\n");
  fprintf(f, "   Leading and trailing spaces   \n");
  fprintf(f, "\n");
  fprintf(f, "Empty lines above and below.\n");
  fprintf(f, "\n");
  fprintf(f, "Final line with MixedCase Words.\n");

  fclose(f);
}

void test_file_normalization(const char* input_file) {
  printf("=== File Normalization Test ===\n");
  
  if (!input_file) {
    printf("No input file provided!\n");
    printf("Usage: ./test <input_file>\n");
    return;
  }
  
  FILE* test_file = fopen(input_file, "r");
  if (!test_file) {
    printf("Error: Cannot open input file '%s'\n", input_file);
    return;
  }
  fclose(test_file);
  
  clock_t start = clock();
  int line_count = normalize_file(input_file, "normalized_output.txt");
  clock_t end = clock();
  
  if (line_count >= 0) {
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Input file: %s\n", input_file);
    printf("Output file: normalized_output.txt\n");
    printf("Processed %d lines in %.6f seconds\n", line_count, time_taken);
    
    if (line_count > 0) {
      double lines_per_sec = line_count / time_taken;
      printf("Processing rate: %.0f lines/second\n", lines_per_sec);
    }
    
    printf("Normalization completed successfully!\n");
  } else {
    printf("File normalization failed!\n");
  }
  printf("\n");
}

void benchmark_normalization() {
  printf("=== Performance Benchmark ===\n");
  
  const char* test_text = "This is a LONGER test string with MULTIPLE words and    SPACES to benchmark the normalization performance of our implementation.";
  const int iterations = 100000;
  
  NormalizedText* nt = create_normalized_text(1024);
  
  clock_t start = clock();
  for (int i = 0; i < iterations; i++) {
    normalize_text_fast(test_text, nt);
  }
  clock_t end = clock();
  
  double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Normalized %d strings in %.6f seconds\n", iterations, time_taken);
  printf("Rate: %.0f normalizations/second\n", iterations / time_taken);
  printf("Sample output: '%s'\n", nt->data);
  
  free_normalized_text(nt);
  printf("\n");
}

int main(int argc, char* argv[]) {
  printf("Text Normalizer Test Suite\n");
  printf("==========================\n\n");
  
  test_basic_normalization();
  test_line_normalization();
  test_batch_normalization();
  
  const char* input_file = (argc > 1) ? argv[1] : NULL;
  test_file_normalization(input_file);
  
  benchmark_normalization();
  
  printf("All tests completed!\n");
  if (input_file) {
    printf("Check 'normalized_output.txt' for your file normalization results.\n");
  }
  
  return 0;
}
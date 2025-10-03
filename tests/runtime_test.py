import time
import tiktoken
import shred

shred_tok = shred.load_encoding("base_50k")
tiktoken_tok = tiktoken.get_encoding("gpt2")

text = "Hello world! This is a runtime comparison test between Shred and tiktoken tokenizers. " * 1000

def benchmark(func, name, iters=5):
  start = time.perf_counter()
  for _ in range(iters):
    func()
  end = time.perf_counter()
  print(f"{name:<20}: {(end - start):.4f} sec for {iters} runs")

print("\n--- Encoding Benchmark ---")
benchmark(lambda: shred_tok.encode(text), "Shred encode")
benchmark(lambda: tiktoken_tok.encode(text), "TikToken encode")

print("\n--- Decoding Benchmark ---")
shred_tokens = shred_tok.encode(text)
tik_tokens = tiktoken_tok.encode(text)

benchmark(lambda: shred_tok.decode(shred_tokens), "Shred decode")
benchmark(lambda: tiktoken_tok.decode(tik_tokens), "TikToken decode")
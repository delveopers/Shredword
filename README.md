# Shredword

A fast and efficient tokenizer library for natural language processing tasks, built with Python and optimized C backend.

## Features

- **High Performance**: Fast tokenization powered by optimized C libraries
- **Multiple Encodings**: Support for various tokenization models and vocabularies
- **Flexible API**: Easy-to-use Python interface with comprehensive functionality
- **Special Tokens**: Built-in support for special tokens and custom vocabularies
- **Fallback Mechanisms**: Robust error handling with fallback tokenization
- **BPE Support**: Byte Pair Encoding implementation for subword tokenization

## Installation

```bash
pip install shredword
```

## Quick Start

```python
from shred import load_encoding

# Load a tokenizer
tokenizer = load_encoding("pre_16k")

# Encode text to tokens
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [10478, 10408, 10416, 10416, ...

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Get vocabulary information
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens}")
```

## Documentation

For detailed usage instructions, API reference, and examples, please see our [User Documentation](https://devsorg.vercel.app/docs/Shredword/User.md).

## Supported Encodings

Shredword supports various pre-trained tokenization models. The library automatically downloads vocabulary files from the official repository when needed.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Clone the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest`

### Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PRs

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/delveopers/shredword/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/delveopers/shredword/discussions)

## Acknowledgments

Built with performance and simplicity in mind for the NLP community.

---

**Note**: This library requires a C/CPP compiler for optimal performance. Fallback Python implementations are available when C/CPP extensions are not available.

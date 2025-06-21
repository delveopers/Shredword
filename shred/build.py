"""
  Build script for the BPE trainer C++ shared library.
  This script compiles the existing C++ files without creating any missing headers.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_compiler():
  """Check if a C++ compiler is available."""
  compilers = ['g++', 'clang++']
  if platform.system() == 'Windows':
    compilers = ['g++', 'clang++', 'cl']
  
  for compiler in compilers:
    try:
      result = subprocess.run([compiler, '--version'], capture_output=True, text=True, timeout=10)
      if result.returncode == 0:
        print(f"Found compiler: {compiler}")
        return compiler
    except (subprocess.TimeoutExpired, FileNotFoundError):
      continue
  return None

from pathlib import Path

def find_source_files():
  """Find the required C++ source files based on the ../src layout."""
  base_dir = Path(__file__).resolve().parent
  bpe_dir = (base_dir / "../src").resolve()
  required_files = [
    bpe_dir / "core.cpp",
    bpe_dir / "hash.cpp",
    bpe_dir / "token.cpp"
  ]

  if all(f.exists() for f in required_files):
    print(f"Found source files in {bpe_dir}")
    return [str(f) for f in required_files]
  else:
    print("[ERROR] Some source files are missing!")
    for f in required_files:
      if not f.exists():
        print(f"  Missing: {f}")
    return None

def build_library():
  """Build the shared library from existing source files."""
  compiler = check_compiler()
  if not compiler:
    print("Error: No C++ compiler found!")
    print("Please install one of the following:")
    print("  - Windows: MinGW-w64 or Visual Studio with C++")
    print("  - Linux: g++ (sudo apt install g++ or equivalent)")
    print("  - macOS: Xcode command line tools (xcode-select --install)")
    return False

  source_files = find_source_files()
  if not source_files:
    print("Error: Could not find required C++ source files!")
    print("Expected files:")
    print("src/")
    print("  - core.cpp, hash.cpp, token.cpp")
    print("  (or similar structure in csrc/ or src/)")
    return False

  # Create build directory
  os.makedirs('build', exist_ok=True)
  
  # Determine library name and compile command based on platform
  if platform.system() == 'Windows':
    lib_name = 'libtoken.dll'
    if compiler == 'cl':
      compile_cmd = [
        'cl', '/LD', '/std:c++11', '/EHsc',
        *source_files,
  '-static-libstdc++', '-static-libgcc',
        f'/Fe:build/{lib_name}'
      ]
    else:
      compile_cmd = [
        compiler, '-shared', '-std=c++11', '-O2',
        *source_files,
  '-static-libstdc++', '-static-libgcc',
        '-o', f'build/{lib_name}',
        '-lregex'
      ]
  elif platform.system() == 'Darwin':
    lib_name = 'libtoken.dylib'
    compile_cmd = [
      compiler, '-shared', '-fPIC', '-std=c++11', '-O2',
      *source_files,
  '-static-libstdc++', '-static-libgcc',
      '-o', f'build/{lib_name}',
      '-lregex'
    ]
  else:  # Linux
    lib_name = 'libtoken.so'
    compile_cmd = [
      compiler, '-shared', '-fPIC', '-std=c++11', '-O2',
      *source_files,
  '-static-libstdc++', '-static-libgcc',
      '-o', f'build/{lib_name}',
      '-lregex'
    ]

  print(f"Building with command: {' '.join(compile_cmd)}")
  
  try:
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
      print(f"Successfully built {lib_name}")
      
      # Copy to project root and other common locations for easier access
      destinations = [
        lib_name,  # Project root
        f'../build/{lib_name}',  # Already there
      ]
      
      for dest in destinations:
        if dest != f'../build/{lib_name}':  # Don't copy to itself
          try:
            shutil.copy(f'build/{lib_name}', dest)
            print(f"Copied to: {dest}")
          except Exception as e:
            print(f"Warning: Could not copy to {dest}: {e}")
      
      return True
    else:
      print("Compilation failed!")
      if result.stdout:
        print("STDOUT:", result.stdout)
      if result.stderr:
        print("STDERR:", result.stderr)
      return False
  except subprocess.TimeoutExpired:
    print("Compilation timed out!")
    return False
  except Exception as e:
    print(f"Compilation error: {e}")
    return False

def verify_build():
  """Verify that the built library can be loaded."""
  import ctypes
  
  # Try to load the library
  lib_files = [
    'libtoken.dll',
    'libtoken.so', 
    'libtoken.dylib',
    '../build/libtoken.dll',
    '../build/libtoken.so',
    '../build/libtoken.dylib'
  ]
  
  for lib_file in lib_files:
    if os.path.exists(lib_file):
      lib_file = os.path.join(os.path.dirname(__file__), "../build/libtoken.dll")
      print("libfile path: ", lib_file)
      try:
        lib = ctypes.CDLL(lib_file)
        print(f"// Successfully verified library: {lib_file}")
        return True
      except Exception as e:
        print(f"X Failed to load {lib_file}: {e}")
  
  return False

def main():
  print("BPE Trainer Build Script")
  print("=" * 30)
  
  # Check if we're in the right directory
  if not (os.path.exists('../src') or os.path.exists('../csrc') or os.path.exists('../src')):
    print("Warning: No source directories found. Make sure you're in the project root.")

  print("Building shared library...")
  success = build_library()
  
  if success:
    print("\nVerifying build...")
    if verify_build():
      print("\n// Build completed successfully!")
      print("You can now run your Python script.")
    else:
      print("\nX Build verification failed.")
      success = False
  else:
    print("\nX Build failed. Please check the error messages above.")
  
  return success

if __name__ == '__main__':
  success = main()
  sys.exit(0 if success else 1)
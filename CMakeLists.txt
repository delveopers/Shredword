cmake_minimum_required(VERSION 3.18)

# Force Ninja generator on Windows if using MinGW
if(WIN32 AND NOT DEFINED CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Ninja")
endif()
project(shredword LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For MinGW on Windows
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Collect all source files
file(GLOB_RECURSE SRC_FILES "src/*.c" "src/*.cpp")
message(STATUS "Found source files: ${SRC_FILES}")
if(NOT SRC_FILES)
  message(FATAL_ERROR "No source files found in src/")
endif()

# Create the shared library target
add_library(token SHARED ${SRC_FILES})
target_include_directories(token PRIVATE shred)
target_link_libraries(token PRIVATE Python::Module)

# Link regex library on Windows with MinGW
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(token PRIVATE regex)
endif()

# Set proper output name and properties for cross-platform compatibility
if(WIN32)
  set_target_properties(token PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(token PROPERTIES PREFIX "lib")
endif()

# Install the shared library (.pyd/.so) directly into shred/
install(TARGETS token DESTINATION . COMPONENT python_modules)
# Install Python files (like cbase.py) into shred/ without nesting
install(DIRECTORY shred/ DESTINATION . COMPONENT python_modules FILES_MATCHING PATTERN "*.py")
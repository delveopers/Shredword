cmake_minimum_required(VERSION 3.18)

# Force Ninja generator on Windows
if(WIN32)
  set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "")
endif()

project(shred LANGUAGES C CXX)

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

# Create the shared library with correct name
add_library(shred SHARED ${SRC_FILES})
target_link_libraries(shred PRIVATE Python::Module)

# Link regex library on Windows with MinGW
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(shred PRIVATE regex)
endif()

# Set proper output name and properties
if(WIN32)
  set_target_properties(shred PROPERTIES 
    OUTPUT_NAME "tokenizer"
    SUFFIX ".pyd"
  )
else()
  set_target_properties(shred PROPERTIES 
    OUTPUT_NAME "tokenizer"
    PREFIX ""
    SUFFIX ".so"
  )
endif()

# Install the compiled extension to the shred package directory
install(TARGETS shred 
  DESTINATION shred 
  COMPONENT python_modules
)

# Install Python files from shred directory
install(DIRECTORY shred/ 
  DESTINATION shred 
  COMPONENT python_modules 
  FILES_MATCHING PATTERN "*.py"
)
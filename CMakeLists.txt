cmake_minimum_required(VERSION 3.18)

project(shredword LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

file(GLOB_RECURSE SRC_FILES "shred/csrc/*.c" "shred/csrc/*.cpp")
message(STATUS "Found source files: ${SRC_FILES}")
if(NOT SRC_FILES)
  message(FATAL_ERROR "No source files found in shred/csrc/")
endif()

add_library(token SHARED ${SRC_FILES})
target_include_directories(token PRIVATE shred/csrc)
target_link_libraries(token PRIVATE Python::Module)

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(token PRIVATE regex)
endif()

if(WIN32)
  set_target_properties(token PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(token PROPERTIES PREFIX "lib")
endif()

install(TARGETS token DESTINATION . COMPONENT python_modules)
install(DIRECTORY shred/ DESTINATION . COMPONENT python_modules FILES_MATCHING PATTERN "*.py")
cmake_minimum_required(VERSION 3.15)
project(pyjags LANGUAGES CXX)

# Find Python and NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 REQUIRED)

# Find JAGS (optional for development)
find_path(JAGS_INCLUDE_DIR Console.h
    PATH_SUFFIXES JAGS
    PATHS 
        /opt/homebrew/opt/jags/include
        /usr/local/include
        /opt/homebrew/include
        /usr/include
)

find_library(JAGS_LIBRARY NAMES JAGS
    PATHS
        /opt/homebrew/opt/jags/lib
        /usr/local/lib
        /opt/homebrew/lib
        /usr/lib
)


# Option to build without JAGS for development
option(BUILD_WITHOUT_JAGS "Build without JAGS dependency for development" OFF)

if(BUILD_WITHOUT_JAGS)
    message(STATUS "Building without JAGS dependency (development mode)")
    # Create a stub extension
    pybind11_add_module(console src/pyjags/console.cpp)
    target_compile_definitions(console PRIVATE NO_JAGS)
elseif(NOT JAGS_INCLUDE_DIR OR NOT JAGS_LIBRARY)
    message(STATUS "JAGS not found. Searched in:")
    message(STATUS "  Headers: /homebrew/opt/jags/include, /usr/local/include, /opt/homebrew/include, /usr/include")
    message(STATUS "  Libraries: /homebrew/opt/jags/lib, /usr/local/lib, /opt/homebrew/lib, /usr/lib")
    message(STATUS "")
    message(STATUS "To install JAGS on macOS:")
    message(STATUS "  brew install jags")
    message(STATUS "")
    message(STATUS "Or build without JAGS for development:")
    message(STATUS "  cmake -DBUILD_WITHOUT_JAGS=ON .")
    message(FATAL_ERROR "Could not find JAGS headers or library")
else()
    message(STATUS "Found JAGS:")
    message(STATUS "  Include: ${JAGS_INCLUDE_DIR}")
    message(STATUS "  Library: ${JAGS_LIBRARY}")
    
    # Build extension with JAGS
    pybind11_add_module(console src/pyjags/console.cpp)
    target_include_directories(console PRIVATE ${JAGS_INCLUDE_DIR})
    target_link_libraries(console PRIVATE ${JAGS_LIBRARY} Python3::NumPy)
endif()

# Install the extension to the correct location
install(TARGETS console DESTINATION pyjags)

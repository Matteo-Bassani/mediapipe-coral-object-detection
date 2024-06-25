cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(
        CMAKE_TOOLCHAIN_FILE 
        ${CMAKE_CURRENT_SOURCE_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake
    )
endif()

# Name the app
project(coralmicro-app)

# Set language requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Coral Micro SDK
include_directories(coralmicro)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../exported_models/apple_detector/)
add_subdirectory(coralmicro)

# Add the source code
add_executable_m7(
    coralmicro-app
    main.cc
    DATA
    ${CMAKE_CURRENT_SOURCE_DIR}/../../exported_models/apple_detector/model_int8_edgetpu.tflite
)

# Link to libraries
target_link_libraries(
    coralmicro-app
    libs_base-m7_freertos
    libs_jpeg_m7
)

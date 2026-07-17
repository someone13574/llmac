cmake_minimum_required(VERSION 3.25)
project(llm-ac-compression LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

include(cmake/llama.cmake)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(
      LLMAC_SANITIZERS
      "Enable AddressSanitizer and UBSan"
      OFF
    )

    if(MSVC)
        add_compile_options(
            /W4
            /permissive-
            /WX
        )
    else()
        add_compile_options(
            -Wall
            -Wextra
            -Wpedantic
            -Wdeprecated
            -Wcast-align
            -Wconversion
            -Wsign-conversion
            -Wdouble-promotion
            -Wimplicit-fallthrough
            -Wmisleading-indentation
            -Wmissing-field-initializers
            -Wnull-dereference
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Woverloaded-virtual
            -Werror
            -Wno-error=unused-function
            -Wno-error=unused-parameter
            -Wno-error=unused-variable
        )

        if(LLMAC_SANITIZERS)
            add_compile_options(
                -fsanitize=address,undefined
                -fno-omit-frame-pointer
            )
            add_link_options(-fsanitize=address,undefined)
        endif()
    endif()
endif()

add_subdirectory(src)

include(CheckCXXCompilerFlag)

add_executable(llmac main.cpp ac.cpp probs.cpp)
target_compile_features(llmac PRIVATE cxx_std_23)
target_link_libraries(llmac PRIVATE llama llama-common)

if(NOT MSVC)
    check_cxx_compiler_flag(-mfma LLMAC_HAS_MFMA)
    if(LLMAC_HAS_MFMA)
        target_compile_options(llmac PRIVATE -mfma)
    endif()

    check_cxx_compiler_flag(-ffp-contract=off LLMAC_HAS_FP_CONTRACT)
    if(LLMAC_HAS_FP_CONTRACT)
        target_compile_options(llmac PRIVATE -ffp-contract=off)
    endif()
endif()

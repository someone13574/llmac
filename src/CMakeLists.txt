add_executable(llmac main.cpp ac.cpp probs.cpp canon.cpp)
target_compile_features(llmac PRIVATE cxx_std_23)
target_link_libraries(llmac PRIVATE llama llama-common)

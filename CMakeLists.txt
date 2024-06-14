cmake_minimum_required(VERSION 3.23)

project(melee)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_STANDARD 20)

include_directories(include)
add_subdirectory(third_party/hnswlib)
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/oneTBB)
add_subdirectory(third_party/cnpy)

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_OPT_LEVEL "avx512")
add_subdirectory(third_party/faiss)

include_directories(third_party/spdlog/include)
include_directories(third_party/oneTBB/include)
include_directories(third_party/cnpy/include)

add_executable(hnsw_build src/hnsw_build.cpp)
target_link_libraries(hnsw_build PRIVATE hnswlib spdlog::spdlog TBB::tbb)

add_executable(hnsw_bench src/hnsw_bench.cpp)
target_link_libraries(hnsw_bench PRIVATE hnswlib spdlog::spdlog TBB::tbb)

add_executable(mem_bench src/mem_bench.cpp)
target_link_libraries(mem_bench PRIVATE spdlog::spdlog TBB::tbb)

add_executable(dist_bench src/dist_bench.cpp)
target_link_libraries(dist_bench PRIVATE spdlog::spdlog TBB::tbb)

add_executable(nnd_build src/nnd_build.cpp)
target_link_libraries(nnd_build PRIVATE faiss spdlog::spdlog TBB::tbb cnpy)
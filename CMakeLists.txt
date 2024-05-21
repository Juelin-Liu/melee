cmake_minimum_required(VERSION 3.16)

project(melee)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_subdirectory(third_party/hnswlib)
add_subdirectory(third_party/spdlog)
add_subdirectory(third_party/oneTBB)
include_directories(third_party/spdlog/include)
include_directories(third_party/oneTBB/include)
add_executable(build src/hnsw_build.cpp)
target_link_libraries(build PRIVATE hnswlib spdlog::spdlog TBB::tbb)

add_executable(bench src/hnsw_bench.cpp)
target_link_libraries(bench PRIVATE hnswlib spdlog::spdlog TBB::tbb)

#add_executable(main src/hnsw_main.cc)
#target_link_libraries(main PRIVATE hnswlib spdlog::spdlog TBB::tbb)
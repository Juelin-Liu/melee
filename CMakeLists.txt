cmake_minimum_required(VERSION 3.20)

project(melee)

find_package(spdlog REQUIRED)
find_package(TBB REQUIRED)

include_directories(include)

add_subdirectory(third_party/hnswlib)

add_executable(main src/hnsw_main.cc)

target_link_libraries(main PRIVATE hnswlib spdlog::spdlog TBB::tbb)
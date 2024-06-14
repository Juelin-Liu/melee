CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

set(CMAKE_CXX_STANDARD 17)
include_directories("include")
option(ENABLE_STATIC "Build static (.a) library" ON)

add_library(cnpy_mmap SHARED "./src/cnpy_mmap.cpp")
install(TARGETS "cnpy_mmap" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(ENABLE_STATIC)
    add_library(cnpy_mmap_static STATIC "./src/cnpy_mmap.cpp")
    set_target_properties(cnpy_mmap_static PROPERTIES OUTPUT_NAME "cnpy_mmap")
    install(TARGETS "cnpy_mmap_static" ARCHIVE DESTINATION lib)
endif(ENABLE_STATIC)

install(FILES "./include/cnpy_mmap.h" DESTINATION include)

find_package(ZLIB)
if (ZLIB_FOUND)
    include_directories(${ZLIB_INCLUDE_DIRS})
    add_library(cnpy SHARED "./src/cnpy.cpp")
    target_link_libraries(cnpy ${ZLIB_LIBRARIES})
    install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    if(ENABLE_STATIC)
        add_library(cnpy-static STATIC "./src/cnpy.cpp")
        set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
        install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
    endif(ENABLE_STATIC)

    install(FILES "./include/cnpy.h" DESTINATION include)
endif()

# add_executable(exp2 "./src/example2.cpp")
# target_link_libraries(exp2 cnpy_mmap)
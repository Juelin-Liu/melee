#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <memory>

namespace melee {
// Function to check the file extension

    enum class DataType {
        Uint8 = 0,
        Int8 = 1,
        Float32 = 2,
        Float16 = 3,
    };

    struct MemmapContainer {
        int _fd{-1};
        void *_container{nullptr};
        size_t _file_bytes{0};
        size_t _header_bytes{0};

        MemmapContainer() = default;
        void * Data() {return (char *) _container + _header_bytes;};
        void Init(int fd, void *start, size_t file_bytes, size_t header_bytes) {
            _fd = fd;
            _container = start;
            _file_bytes = file_bytes;
            _header_bytes = header_bytes;
        };

        MemmapContainer(int fd, void *file_data, size_t file_bytes, size_t header_bytes){
            Init(fd, file_data, file_bytes, header_bytes);
        };

        ~MemmapContainer() {
            if (_container) {
                munmap(_container, _file_bytes);
            }
            if (_fd != -1) close(_fd);
        };
    };



    struct Matrix2D {
        std::shared_ptr<MemmapContainer> mmap_data{nullptr};
        size_t shape[2]{0, 0};
        size_t word_size{0};
        DataType dtype;

        void *get_vec(size_t row_id) {
            return (char *) mmap_data->Data() + row_id * shape[1] * word_size;
        }

        template<typename T>
        T* data() {
            return static_cast<T *>(mmap_data->Data());
        }
    };

    std::string getFileExtension(const std::string &filename) {
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::runtime_error("File has no extension.");
        }
        return filename.substr(dotPos + 1);
    };

    Matrix2D loadMatrix(const std::string &filename) {
        // Open the binary file
        Matrix2D ret;
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Get the file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            throw std::runtime_error("Cannot get file size: " + filename);
        }

        // Memory-map the file
        void *fileData = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (fileData == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Memory mapping failed: " + filename);
        }

        // Read number of rows and columns
        uint32_t rows, cols;
        std::memcpy(&rows, fileData, sizeof(rows));
        std::memcpy(&cols, static_cast<char *>(fileData) + sizeof(rows), sizeof(cols));
        ret.mmap_data = std::make_shared<MemmapContainer>(MemmapContainer(fd, fileData, sb.st_size, 2 * sizeof(uint32_t)));
        ret.shape[0] = rows;
        ret.shape[1] = cols;
        // Determine the type of the matrix elements from the file extension
        std::string extension = getFileExtension(filename);

        if (extension == "fbin") {
            ret.word_size = 4;
            ret.dtype = DataType::Float32;
        } else if (extension == "u8bin") {
            ret.word_size = 1;
            ret.dtype = DataType::Uint8;
        } else if (extension == "i8bin"){
            ret.word_size = 1;
            ret.dtype = DataType::Int8;
        } else if (extension == "f16bin") {
            ret.word_size = 2;
            ret.dtype = DataType::Float16;
        }
        return ret;
    }
}
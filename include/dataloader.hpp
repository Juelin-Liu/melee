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

enum class DataType
{
    Uint8 = 0,
    Int8 = 1,
    Float32 = 2,
    Float16 = 3,
}

struct Matrix2D
{
    void *data{nullptr};
    size_t dim{0};
    size_t row{0};
    size_t word_size{0};
    bool use_mmap{true};
    DataType dtype;
    void* get_vec(size_t row_id) {
        return data + dim * row * work_size;
    }

    ~Matrix2D() {
        if (use_mmap && data) {

        } else if (data) {
            
        }
    }
}

using Matrix2DPtr = std::shared_ptr<Matrix2D>;

std::string getFileExtension(const std::string &filename) {
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::runtime_error("File has no extension.");
    }
    return filename.substr(dotPos + 1);
};

Matrix2DPtr loadMatrix(const std::string &filename) {
    // Open the binary file

    auto ret = std::make_shared<Matrix2D>();

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

    close(fd);

    // Read number of rows and columns
    uint32_t rows, cols;
    std::memcpy(&rows, fileData, sizeof(rows));
    std::memcpy(&cols, static_cast<char*>(fileData) + sizeof(rows), sizeof(cols));

    ret->row = rows;
    ret->dim = cols;
    ret->data = static_cast<char *>(fileData) + sizeof(rows) + sizeof(cols);

    // Determine the type of the matrix elements from the file extension
    std::string extension = getFileExtension(filename);

    if (extension == "fbin") {
        floatMatrix.resize(rows, std::vector<float>(cols));
        const float *matrixData = static_cast<const float*>(static_cast<const void*>(static_cast<const char*>(fileData) + 2 * sizeof(uint32_t)));
        for (uint32_t i = 0; i < rows; ++i) {
            std::memcpy(floatMatrix[i].data(), matrixData + i * cols, cols * sizeof(float));
        }
    } else if (extension == "u8bin") {
        u8Matrix.resize(rows, std::vector<uint8_t>(cols));
        const uint8_t *matrixData = static_cast<const uint8_t*>(static_cast<const void*>(static_cast<const char*>(fileData) + 2 * sizeof(uint32_t)));
        for (uint32_t i = 0; i < rows; ++i) {
            std::memcpy(u8Matrix[i].data(), matrixData + i * cols, cols * sizeof(uint8_t));
        }
    } else {
        munmap(fileData, sb.st_size);
        throw std::runtime_error("Unsupported file extension: " + extension);
    }

    // Unmap the file
    munmap(fileData, sb.st_size);
}
}
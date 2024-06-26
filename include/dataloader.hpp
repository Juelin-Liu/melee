#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace melee {
// Function to check the file extension

enum class DataType {
  Uint8 = 0,
  Int8 = 1,
  Float32 = 2,
  Float16 = 3,
};

struct Matrix2D {
  void *_data{nullptr};
  size_t shape[2]{0, 0};
  size_t word_size{0};
  DataType dtype;

  void *get_vec(size_t row_id) {
    return static_cast<char *>(_data) + row_id * shape[1] * word_size;
  }

  template <typename T> T *data() { return static_cast<T *>(_data); }
};

struct GroundTruth {
  std::vector<uint32_t > _label;
  std::vector<float> _distance;
  size_t shape[2]{0, 0};
};

std::string getFileExtension(const std::string &filename) {
  size_t dotPos = filename.find_last_of('.');
  if (dotPos == std::string::npos) {
    throw std::runtime_error("File has no extension.");
  }
  return filename.substr(dotPos + 1);
};

inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

GroundTruth loadGT(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {};
  }

  // Open the binary file
  GroundTruth ret;
  // Read number of rows and columns
  uint32_t rows, cols;
  // Read rows and cols
  file.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
  ret.shape[0] = rows;
  ret.shape[1] = cols;
  size_t num_elements = ret.shape[0] * ret.shape[1];
  ret._label.resize(num_elements);
  ret._distance.resize(num_elements);
  file.read(reinterpret_cast<char *>(ret._label.data()), num_elements * sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(ret._distance.data()), num_elements * sizeof(float));
  file.close();
  return ret;
};

Matrix2D loadMatrix(const std::string &filename, uint32_t max_elements, bool to_float=false) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    exit(-1);
  }

  // Open the binary file
  Matrix2D ret;
  // Read number of rows and columns
  uint32_t rows, cols;
  // Read rows and cols
  file.read(reinterpret_cast<char *>(&rows), sizeof(int));
  file.read(reinterpret_cast<char *>(&cols), sizeof(int));
  rows = std::min(rows, max_elements);
  ret.shape[0] = rows;
  ret.shape[1] = cols;
  // Determine the type of the matrix elements from the file extension
  std::string extension = getFileExtension(filename);
  if (to_float) {
      ret.word_size = 4;
      ret.dtype = DataType::Float32;
      if (extension == "fbin") {
          size_t num_bytes = ret.word_size * ret.shape[0] * ret.shape[1];
          ret._data = new unsigned char[num_bytes];
          file.read(ret.data<char>(), num_bytes);
          file.close();
          return ret;
      } else if (extension == "u8bin") {
          size_t arr_size = ret.shape[0] * ret.shape[1];
          std::vector<uint8_t > data(arr_size);
          file.read((char *)data.data(), arr_size * sizeof(uint8_t));
          file.close();

          size_t num_bytes = ret.word_size * ret.shape[0] * ret.shape[1];
          ret._data = new unsigned char[num_bytes];
          auto *ptr = reinterpret_cast<float*>(ret._data);
          for (size_t i = 0; i < data.size(); i++){
              ptr[i] = static_cast<float>(data.at(i));
          }
          return ret;
      } else if (extension == "i8bin") {
          size_t arr_size = ret.shape[0] * ret.shape[1];
          std::vector<int8_t > data(arr_size);
          file.read((char *)data.data(), arr_size * sizeof(int8_t));
          file.close();

          size_t num_bytes = ret.word_size * ret.shape[0] * ret.shape[1];
          ret._data = new unsigned char[num_bytes];
          auto *ptr = reinterpret_cast<float*>(ret._data);
          for (size_t i = 0; i < data.size(); i++){
              ptr[i] = static_cast<float>(data.at(i));
          }
          return ret;
      } else if (extension == "f16bin") {
          size_t arr_size = ret.shape[0] * ret.shape[1];
          std::vector<_Float16> data(arr_size);
          file.read((char *)data.data(), arr_size * sizeof(_Float16));
          file.close();

          size_t num_bytes = ret.word_size * ret.shape[0] * ret.shape[1];
          ret._data = new unsigned char[num_bytes];
          auto *ptr = reinterpret_cast<float*>(ret._data);
          for (size_t i = 0; i < data.size(); i++){
              ptr[i] = static_cast<float>(data.at(i));
          }
          return ret;
      }
  } else {
      if (extension == "fbin") {
          ret.word_size = 4;
          ret.dtype = DataType::Float32;
      } else if (extension == "u8bin") {
          ret.word_size = 1;
          ret.dtype = DataType::Uint8;
      } else if (extension == "i8bin") {
          ret.word_size = 1;
          ret.dtype = DataType::Int8;
      } else if (extension == "f16bin") {
          ret.word_size = 2;
          ret.dtype = DataType::Float16;
      }
      size_t num_bytes = ret.word_size * ret.shape[0] * ret.shape[1];
      ret._data = new unsigned char[num_bytes];
      file.read(ret.data<char>(), num_bytes);
      file.close();
      return ret;
  }
}
} // namespace melee
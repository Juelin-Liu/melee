#include "cnpy_mmap.h"
int main(int argc, const char * argv[]) {
    auto input_path = argv[1];
    auto arr = cnpyMmap::npy_load(input_path);
    auto vec = arr.data<int64_t>();
    std::cout << "input vector: ";
    for (size_t i = 0; i < arr.num_vals; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

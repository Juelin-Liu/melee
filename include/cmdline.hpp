#pragma once
#include "argparse.hpp"
#include <thread>

namespace melee {
#define ALWAYS_ASSERT(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "Assertion failed: " #expr << std::endl;                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

    struct MemConfig {
        size_t N; // Number of random reads
        size_t K; // Size of block to read (in bytes)
        size_t ARR_SIZE; // Size of array (in bytes)
        size_t T; // Number of threads
        MemConfig() = default;
        MemConfig(int argc, char * argv[]) {
            Init(argc, argv);
        }

        void Init(int argc, char * argv[]) {
            constexpr size_t default_arr_size = 4ul * 1024ul * 1024ul * 1024ul;
            argparse::ArgumentParser program("Memory benchmark binary");
            program.add_argument("--N").help("number of random reads").scan<'u', size_t>().default_value(100 * 10000ul); // 1M
            program.add_argument("--K").help("size of block to read").scan<'u', size_t>().default_value(128ul); // 128 bytes
            program.add_argument("--T").help("number of threads").scan<'u', size_t>().default_value(1ul); // 1 threads
            program.add_argument("--ARR_SIZE").help("array size").scan<'u', size_t>().default_value(default_arr_size); // 4GB bytes
            program.parse_args(argc, argv);

            N = program.get<size_t>("--N");
            K = program.get<size_t>("--K");
            T = program.get<size_t>("--T");
            ARR_SIZE = program.get<size_t>("--ARR_SIZE");
        }
    };
    struct NNDBuildConfig {
        std::string space;
        size_t M;
        size_t max_elements;
        std::string feat_path;
        std::string index_path;
        NNDBuildConfig() = default;
        NNDBuildConfig(int argc, char* argv[]) {
            Init(argc, argv);
        };
        void Init(int argc, char* argv[]) {
            argparse::ArgumentParser program("NND build index binary");
            program.add_argument("--space").help("one of l2, ip, or cosine").required();
            program.add_argument("--M").help(" maximum number of outgoing connections in the graph").scan<'u', size_t>().default_value(16);
//            program.add_argument("--ef_construction").help("priority queue capacity during the index construction").scan<'u', size_t>().default_value(200);
            program.add_argument("--max_elements").help("max elements in the graph").scan<'u', size_t>().default_value(1000000); // 1M
            program.add_argument("--feat_path").help("path to the feature file").required();
            program.add_argument("--index_path").help("path to the output index").required();
            program.parse_args(argc, argv);
            space =  program.get<std::string>("--space");
            M = program.get<size_t>("--M");
//            ef_construction = program.get<size_t>("--ef_construction");
            max_elements = program.get<size_t>("--max_elements");
            feat_path = program.get<std::string >("--feat_path");
            index_path = program.get<std::string >("--index_path");
        }
    };
    struct HNSWBuildConfig {
        std::string space;
        size_t M;
        size_t ef_construction;
        size_t max_elements;
        std::string feat_path;
        std::string index_path;
        HNSWBuildConfig() = default;
        HNSWBuildConfig(int argc, char* argv[]) {
            Init(argc, argv);
        };

        void Init(int argc, char* argv[]) {
            argparse::ArgumentParser program("HNSW build index binary");
            program.add_argument("--space").help("one of l2, ip, or cosine").required();
            program.add_argument("--M").help(" maximum number of outgoing connections in the graph").scan<'u', size_t>().default_value(16);
            program.add_argument("--ef_construction").help("priority queue capacity during the index construction").scan<'u', size_t>().default_value(200);
            program.add_argument("--max_elements").help("max elements in the graph").scan<'u', size_t>().default_value(1000000); // 1M
            program.add_argument("--feat_path").help("path to the feature file").required();
            program.add_argument("--index_path").help("path to the output index").required();
            program.parse_args(argc, argv);
            space =  program.get<std::string>("--space");
            M = program.get<size_t>("--M");
            ef_construction = program.get<size_t>("--ef_construction");
            max_elements = program.get<size_t>("--max_elements");
            feat_path = program.get<std::string >("--feat_path");
            index_path = program.get<std::string >("--index_path");
        }
    };

    struct HNSWBenchConfig {
        std::string space;
        std::string index_path;
        std::string query_path;
        std::string truth_path;
        size_t num_threads;
        HNSWBenchConfig() = default;
        HNSWBenchConfig(int argc, char* argv[]) {
            Init(argc, argv);
        };

        void Init(int argc, char* argv[]) {
            argparse::ArgumentParser program("HNSW benchmark program");
            program.add_argument("--space").help("one of l2, ip, or cosine").required();
            program.add_argument("--index_path").help("path to the index file").required();
            program.add_argument("--query_path").help("path to the query file").required();
            program.add_argument("--truth_path").help("path to the truth file").required();
            program.add_argument("--num_threads").help("number of threads").scan<'u', size_t>().default_value(std::thread::hardware_concurrency());
            program.parse_args(argc, argv);
            space =  program.get<std::string>("--space");
            index_path = program.get<std::string >("--index_path");
            query_path = program.get<std::string >("--query_path");
            truth_path = program.get<std::string >("--truth_path");
            num_threads = program.get<size_t>("--num_threads");
        }
    };


    struct HNSWConfig {
        // graph construction parameters:
        std::string space; // name of the space (can be one of "l2", "ip", or "cosine").
        size_t dim; // dimensionality of the space.
        size_t M; // parameter that defines the maximum number of outgoing connections in the graph.
        size_t ef_construction; // parameter that controls speed/accuracy trade-off during the index construction.
        size_t max_elements; //  capacity of the index

        // query time parameters:
        size_t ef;
        size_t num_threads;
        size_t k;

        std::string feat_path;
        std::string index_path;
        std::string query_path;
        std::string truth_path;
        std::string index_out;
    };

    HNSWConfig get_hnsw_config(int argc, char *argv[]) {
        HNSWConfig config;
        argparse::ArgumentParser program("HNSW profiler");
        program.add_argument("--space").help("one of l2, ip, or l2uint8").required();
        program.add_argument("--M").help(" maximum number of outgoing connections in the graph").scan<'u', size_t>().default_value(16);
        program.add_argument("--ef_construction").help("priority queue capacity during the index construction").scan<'u', size_t>().default_value(200);
        program.add_argument("--ef").help("priority queue capacity during the index construction").scan<'u', size_t>().default_value(100);
        program.add_argument("--num_threads").help("capacity of the index").scan<'u', size_t>().default_value(1);
        program.add_argument("--k").help("top k search index").scan<'u', size_t>().default_value(10);
        program.add_argument("--max_elements").help("max elements in the graph").scan<'u', size_t>().default_value(1000000); // 1M

        program.add_argument("--feat_path").help("path to the feature file").default_value("");;
        program.add_argument("--index_path").help("path to the graph index file").default_value("");
        program.add_argument("--query_path").help("path to the query feature file").default_value("");
        program.add_argument("--truth_path").help("path to the ground truth file").default_value("");
        program.add_argument("--index_out").help("index output directory").default_value("");
        program.parse_args(argc, argv);

        config.space = program.get<std::string>("--space");
        config.M = program.get<size_t>("--M");
        config.ef_construction = program.get<size_t>("--ef_construction");
        config.ef = program.get<size_t>("--ef");
        config.num_threads = program.get<size_t>("--num_threads");
        config.k = program.get<size_t>("--k");
        config.max_elements = program.get<size_t>("--max_elements");

        config.feat_path = program.get<std::string>("--feat_path");
        config.index_path = program.get<std::string>("--index_path");
        config.query_path = program.get<std::string>("--query_path");
        config.truth_path = program.get<std::string>("--truth_path");
        config.index_out = program.get<std::string>("--index_out");

        return config;
    };
} // namespace profiler

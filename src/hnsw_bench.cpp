#include <hnswlib/hnswlib.h>
#include <numeric>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <queue>
#include <vector>
#include <spdlog/spdlog.h>

#include "cmdline.hpp"
#include "dataloader.hpp"
#include "span.hpp"
#include "timer.hpp"

#define ALWAYS_ASSERT(expr)                                                    \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "Assertion failed: " #expr << std::endl;                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

using namespace melee;


template<typename space_t, typename dist_t>
void bench(BenchConfig config) {
    typedef std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> ResultType;
    std::vector<int> all_k{1, 10, 100};
    std::vector<int> all_ef{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500};
    Matrix2D query;
    GroundTruth truth;

    query = loadMatrix(config.query_path, 1e6);
    truth = loadGT(config.truth_path);
    ALWAYS_ASSERT(query.shape[0] == truth.shape[0]);
    space_t space(query.shape[1]);
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw{nullptr};

    Timer timer;
    timer.start();

    spdlog::info("Loading Index From: {}", config.index_path);
    alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space, config.index_path);
    timer.end();
    spdlog::info("Load Index in {} secs", timer.seconds());


    // search kNN and evaluation
    const int num_queries = query.shape[0];
    spdlog::info("Start Benchmarking");

    for (int k: all_k) {
        for (int ef: all_ef) {
            if (ef < k) continue;

            alg_hnsw->setEf(ef);
            alg_hnsw->metric_hops = 0;
            alg_hnsw->metric_base_hops = 0;
            alg_hnsw->metric_distance_computations = 0;
            alg_hnsw->metric_base_distance_computations = 0;

            int total_matched = 0;
            timer.start();

            std::vector<ResultType> results(num_queries);
            oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<int64_t>(0, num_queries, query.shape[0] / 100),
                    [&](const oneapi::tbb::blocked_range<int64_t> &r) {
                        for (int64_t i = r.begin(); i < r.end(); i++) {
                            results.at(i) = alg_hnsw->searchKnn(query.get_vec(i), k);
                        }
                    }, oneapi::tbb::simple_partitioner());

//            for (size_t i = 0; i < num_queries; i++) {
//                auto search_result = alg_hnsw->searchKnnCloserFirst(query.get_vec(i), k);
//                auto start_label_ptr = truth._label.data() + i * truth.shape[1];
//                auto start_dist_ptr = truth._distance.data() + i * truth.shape[1];
//                auto end_label_ptr = start_label_ptr + k;
//                auto end_dist_ptr = start_dist_ptr + k;
//                std::vector<uint32_t> gt_label(start_label_ptr, end_label_ptr);
//                std::vector<float> gt_distance(start_dist_ptr, end_dist_ptr);
//                for (auto p : search_result) {
//                    auto pred_label = p.second;
//                    auto pred_dist = p.first;
//                    for (auto label: gt_label) {
//                        if (label == pred_label)
//                            total_matched++;
//                    };
//                }
//            }
            timer.end();
            double search_time = timer.seconds();
            // spdlog::info("Queries={}", num_queries);
            // spdlog::info("SearchTime={0:.2f} secs", search_time);
            // spdlog::info("QPS={0:.2f}", 1.0 * num_queries / search_time);

            for (int i = 0; i < num_queries; i++) {
                auto search_result = results.at(i);
                while (!search_result.empty()) {
                    auto p = search_result.top();
                    auto pred_label = p.second;
                    auto pred_dist = p.first;
                    search_result.pop();
                    for (int j = 0; j < k; j++) {
                        auto idx = i * truth.shape[1] + j;
                        auto t = truth._label[idx];
                        if (t == pred_label)
                            total_matched++;
                    };
                }
            }

            double recall = 100.0 * total_matched / (k * num_queries);
            // spdlog::info("Recall={0:.2f}%", recall);

            long num_upper_hops = alg_hnsw->metric_hops;
            long num_base_hops = alg_hnsw->metric_base_hops;
            long num_total_hops = num_upper_hops + num_base_hops;

            long num_upper_dist = alg_hnsw->metric_distance_computations;
            long num_base_dist = alg_hnsw->metric_base_distance_computations;
            long num_total_dist = num_upper_dist + num_base_dist;

            float upper_memory = 1.0 * num_upper_dist * query.shape[1] * query.word_size / 1e6;
            float base_memory = 1.0 * num_base_dist * query.shape[1] * query.word_size / 1e6;
            float total_memory = upper_memory + base_memory;

            // spdlog::info("UpperHops={}", num_upper_hops);
            // spdlog::info("BaseHops={}", num_base_hops);

            // spdlog::info("UpperDist={}", num_upper_dist);
            // spdlog::info("BaseDist={}", num_base_dist);

            // spdlog::info("UpperRead={0:.2f} MB", upper_memory);
            // spdlog::info("TotalRead={0:.2f} MB", base_memory);
            // spdlog::info("Throughput={0:.2f} MB/s", total_memory / search_time);

            // spdlog::info("PerQueryHops={0:.2f}", 1.0 * num_total_hops / query.shape[0]);
            // spdlog::info("PerQueryDist={0:.2f}", 1.0 * num_total_dist / query.shape[0]);
            // spdlog::info("PerQueryRead={0:.2f} MB", total_memory / query.shape[0]);

            spdlog::info("k={} ef={} thops={} bhops={} tmem={:.3f}MB qps={} recall={:.1f}", k, ef, num_upper_hops / num_queries, num_base_hops / num_queries, total_memory / num_queries, int(num_queries / search_time), recall);
        }
    }
    spdlog::info("End Benchmarking");
}

int main(int argc, char *argv[]) {
    BenchConfig config;
    config.Init(argc, argv);
    if (config.space == "ip") {
        bench<hnswlib::InnerProductSpace, float>(config);
    } else if (config.space == "l2") {
        if (ends_with(config.query_path, "fbin")) {
            bench<hnswlib::L2Space, float>(config);
        } else if (ends_with(config.query_path, "u8bin")) {
            bench<hnswlib::L2SpaceI, int>(config);
        }
    } else {
        spdlog::error("Invalid space {}", config.space);
    }
}
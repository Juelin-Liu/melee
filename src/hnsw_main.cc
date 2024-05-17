#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>
#include <numeric>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>

#include "cmdline.hpp"
#include "timer.hpp"
#include "dataloader.hpp"
#include "span.hpp"
using namespace melee;
template<typename space_t, typename dist_t>
void benchmark(HNSWConfig config){
    typedef std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> ResultType;

    // TODO feat, query, truth;
    Matrix2D feat, query, truth;
    feat = loadMatrix(config.feat_path);
    config.max_elements = feat.shape[0];
    config.dim = feat.shape[1];
    
    if (config.query_path.empty() != config.truth_path.empty()){
        spdlog::error("query and ground truth should be both provided (or not provided)");
        exit(-1);
    }

    if (!config.query_path.empty())
    {
        query = loadMatrix(config.query_path);
        if (feat.shape[1] != query.shape[1]) {
            spdlog::error("feature dimension {} and query dimension {} doesn't match", feat.shape[1], query.shape[1]);
            exit(-1);
        }
    }

    if (!config.truth_path.empty())
    {
        truth = loadMatrix(config.truth_path);
        if (query.shape[0] != truth.shape[0]) {
            spdlog::error("query elements {} and ground truth elements {} doesn't match", query.shape[0], truth.shape[0]);
            exit(-1);
        }
        if (config.k > truth.shape[1]) {
            spdlog::error("top k is larger than elements in the ground truth");
            exit(-1);
        }
    }

    space_t space(config.dim);
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw{nullptr};

    Timer timer;
    timer.start();

    if (!config.index_path.empty())
    {
        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space, config.index_path);
        config.max_elements = alg_hnsw->max_elements_;
        config.M = alg_hnsw->M_;
        config.ef_construction = alg_hnsw->ef_construction_;
    }
    else
    {
        alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space,
                                                          config.max_elements,
                                                          config.M,
                                                          config.ef_construction);

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int64_t>(0, config.max_elements),
                                  [&](const oneapi::tbb::blocked_range<int64_t> &r)
                                  {
                                      for (int64_t i = r.begin(); i < r.end(); i++)
                                      {
                                          alg_hnsw->addPoint(feat.get_vec(i), i);
                                      }
                                  });
    }

    timer.end();
    spdlog::info("BuildTime={} secs", timer.seconds());

    if (!config.index_out.empty() && config.index_path.empty())
    {
        spdlog::info("Saving Index To: {}", config.index_out);
        alg_hnsw->saveIndex(config.index_out);
    }

    // search kNN and evaluation
    if (query.shape[0] > 0)
    {
        int64_t num_queries = query.shape[0];
        alg_hnsw->setEf(config.ef);

        std::vector<ResultType> results(num_queries);
        timer.start();
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int64_t>(0, num_queries),
                                  [&](const oneapi::tbb::blocked_range<int64_t> &r)
                                  {
                                      for (int64_t i = r.begin(); i < r.end(); i++)
                                      {
                                          results.at(i) = alg_hnsw->searchKnn(query.get_vec(i), config.k);
                                      }
                                  });
        timer.end();
        double search_time = timer.seconds();
        spdlog::info("SearchTime={0:.2f} secs", search_time);
        spdlog::info("QPS={0:.2f}", num_queries / search_time);

        std::vector<int> matched(num_queries);
        auto ground_truths = span(truth.data<int >(), truth.shape[0] * truth.shape[1]);

        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int64_t>(0, num_queries),
                                  [&](const oneapi::tbb::blocked_range<int64_t> &r)
                                  {
                                      for (int64_t i = r.begin(); i < r.end(); i++)
                                      {
                                          auto search_result = results.at(i);
                                          auto ground_truth = ground_truths.subspan(i * truth.shape[1], config.k);
                                          while (!search_result.empty()){
                                              auto p = search_result.top();
                                              for (auto gt: ground_truth) {
                                                  if (gt == p.second) {
                                                      matched.at(i)++;
                                                      break;
                                                  }
                                              }
                                              search_result.pop();
                                          }
                                      }
                                  });
        int64_t total_matched = std::accumulate(matched.begin(), matched.end(), 0ll);
        double recall = 1.0 * total_matched / (config.k * num_queries);
        spdlog::info("Recall={0:.2f}%", recall * 100);

        long num_upper_hops = alg_hnsw->metric_hops;
        long num_base_hops = alg_hnsw->metric_base_hops;
        long num_total_hops = num_upper_hops + num_base_hops;

        long num_upper_dist = alg_hnsw->metric_distance_computations;
        long num_base_dist = alg_hnsw->metric_base_distance_computations;
        long num_total_dist = num_upper_dist + num_base_dist;

        float upper_memory = 1.0 * num_upper_dist * config.dim * sizeof(dist_t) / 1e6;
        float base_memory = 1.0 * num_base_dist * config.dim * sizeof(dist_t) / 1e6;
        float total_memory = upper_memory + base_memory;

        spdlog::info("UpperHops={}", num_upper_hops);
        spdlog::info("BaseHops={}", num_base_hops);

        spdlog::info("UpperDist={}", num_upper_dist);
        spdlog::info("BaseDist={}", num_base_dist);
        
        spdlog::info("UpperRead={0:.2f} MB", upper_memory);
        spdlog::info("TotalRead={0:.2f} MB", base_memory);
        spdlog::info("Throughput={0:.2f} MB/s", total_memory / search_time);

        spdlog::info("QHops={0:.2f}", 1.0 * num_total_hops / query.shape[0]);
        spdlog::info("QDist={0:.2f}", 1.0 * num_total_dist / query.shape[0]);
        spdlog::info("QRead={0:.2f} MB", total_memory / query.shape[0]);
    }


}

int main(int argc, char *argv[])
{
    HNSWConfig config = get_hnsw_config(argc, argv);
    oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, config.num_threads);
    if (config.space == "ip") {
        benchmark<hnswlib::InnerProductSpace, float>(config);
    } else if (config.space == "l2") {
        benchmark<hnswlib::L2Space, float>(config);
    } else if (config.space == "l2uint8") {
        benchmark<hnswlib::L2SpaceI, int>(config);
    }
}
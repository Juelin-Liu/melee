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

using namespace melee;
template <typename space_t, typename dist_t> void build(HNSWBuildConfig config) {
    typedef std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> ResultType;

    Matrix2D feat = loadMatrix(config.feat_path, config.max_elements);
    space_t space(feat.shape[1]);
    hnswlib::HierarchicalNSW<dist_t> *alg_hnsw{nullptr};
    Timer timer;
    timer.start();
    ALWAYS_ASSERT(!config.feat_path.empty());
    spdlog::info("Loading Data From: {}", config.feat_path);
    spdlog::info("Adding {}M points", config.max_elements / 1e6);

    alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(
            &space, config.max_elements, config.M, config.ef_construction);

    oneapi::tbb::parallel_for(
            oneapi::tbb::blocked_range<size_t>(0, config.max_elements, feat.shape[0] / 1000),
            [&](const oneapi::tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    alg_hnsw->addPoint(feat.get_vec(i), i);
                }
            }, oneapi::tbb::simple_partitioner());

    timer.end();
    spdlog::info("BuildTime={} secs", timer.seconds());
    spdlog::info("Saving Index To: {}", config.index_path);
    alg_hnsw->saveIndex(config.index_path);
}

int main(int argc, char *argv[]) {
    HNSWBuildConfig config;
    config.Init(argc, argv);

    if (config.space == "ip") {
        build<hnswlib::InnerProductSpace, float>(config);
    } else if (config.space == "l2") {
        if (ends_with(config.feat_path, "fbin")) {
            build<hnswlib::L2Space, float>(config);
        } else if (ends_with(config.feat_path, "u8bin")){
            build<hnswlib::L2SpaceI, int>(config);
        }
    } else {
        spdlog::error("Invalid space {}", config.space);
    }
}
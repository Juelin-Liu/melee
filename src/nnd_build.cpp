#include <spdlog/spdlog.h>
#include <faiss/IndexNNDescent.h>
#include <cnpy.h>

#include "cmdline.hpp"
#include "dataloader.hpp"
#include "span.hpp"
#include "timer.hpp"

using namespace melee;
void Build(NNDBuildConfig config){
    auto metric = faiss::MetricType::METRIC_L2;
    if (config.space == "l2") {
        metric = faiss::MetricType::METRIC_L2;
    } else if (config.space == "ip") {
        metric = faiss::MetricType::METRIC_Lp;
    } else if (config.space == "l1") {
        metric = faiss::MetricType::METRIC_L1;
    }

    Matrix2D feat = loadMatrix(config.feat_path, config.max_elements, true);
    Timer timer;
    timer.start();
    size_t d = feat.shape[1];
    size_t K = config.M;
    auto index = std::make_unique<faiss::IndexNNDescentFlat>(d, K, metric);
    index->verbose = true;
    index->add(feat.shape[0], feat.data<float>());

    timer.end();
    auto build_time = timer.seconds();
    spdlog::info("NND build time={:.1f} secs", build_time);
    cnpy::npy_save(config.index_path, index->nndescent.final_graph);
    spdlog::info("Saved nnd index to {}", config.index_path);

};

int main(int argc, char *argv[]) {
    NNDBuildConfig config;
    config.Init(argc, argv);
    Build(config);
}
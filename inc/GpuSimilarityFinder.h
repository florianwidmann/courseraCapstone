#pragma once

#include <tuple>
#include <vector>

#include "SimilarityFinderOptions.h"

namespace GpuSimilarityFinder
{
    void setOptions(const SimilarityFinderOptions &options);

    std::tuple<std::vector<unsigned int>, std::vector<float>, unsigned int> findSimilarities(
        unsigned int tileDim,
        unsigned int blockDim,
        const uint8_t* paddedImage,
        unsigned int paddedHeight,
        unsigned int paddedWidth);
}

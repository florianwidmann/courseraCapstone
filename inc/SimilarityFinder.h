#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "SimilarityFinderOptions.h"

class SimilarityFinder
{
public:
    static constexpr unsigned int blackTileIdx = (unsigned int)-1;

    static SimilarityFinder& getInstance();

    SimilarityFinder(const SimilarityFinder&) = delete;
    SimilarityFinder& operator=(const SimilarityFinder&) = delete;
    SimilarityFinder(SimilarityFinder&&) = delete;
    SimilarityFinder& operator=(SimilarityFinder&&) = delete;

    ~SimilarityFinder() = default;

    void setOptions(const SimilarityFinderOptions& options);

    unsigned int getPaddedSize(unsigned int heightOrWidth) const;

    std::tuple<std::vector<unsigned int>, std::vector<float>, unsigned int, unsigned int> findSimilarities(
        const unsigned int origHeight,
        const unsigned int origWidth,
        const uint8_t *paddedImage,
        unsigned int paddedHeight,
        unsigned int paddedWidth) const;

private:
    std::optional<SimilarityFinderOptions> options_;

    SimilarityFinder() = default;
};

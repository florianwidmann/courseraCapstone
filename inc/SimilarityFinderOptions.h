#pragma once

#include <string>
#include <utility>
#include <vector>

class SimilarityFinderOptions;

namespace GpuSimilarityFinder
{
    void setOptions(const SimilarityFinderOptions &options);
}

// Encapsulates the options for the similarity finder.
class SimilarityFinderOptions
{
public:
    SimilarityFinderOptions() = default;

    SimilarityFinderOptions(const SimilarityFinderOptions& ) = default;
    SimilarityFinderOptions& operator=(const SimilarityFinderOptions& ) = default;
    SimilarityFinderOptions(SimilarityFinderOptions&& ) = default;
    SimilarityFinderOptions& operator=(SimilarityFinderOptions&& ) = default;

    ~SimilarityFinderOptions() = default;

    static std::pair<std::vector<std::string>, SimilarityFinderOptions> parseFromArgLine(int argc, char *argv[]);

    SimilarityFinderOptions& setSuffix(const std::string& suffix);
    SimilarityFinderOptions& setTileDim(unsigned int tileDim);
    SimilarityFinderOptions& setBlockDim(unsigned int blockDim);
    SimilarityFinderOptions& setWindowDim(unsigned int windowDim);
    SimilarityFinderOptions& setEpsilonLow(float epsilonLow);
    SimilarityFinderOptions& setEpsilonHigh(float epsilonHigh);
    SimilarityFinderOptions& setScalarBlue(float scalarBlue);
    SimilarityFinderOptions& setScalarGreen(float scalarGreen);
    SimilarityFinderOptions& setScalarRed(float scalarRed);

    const std::string& getSuffix() const;
    unsigned int getTileDim() const;
    unsigned int getBlockDim() const;
    unsigned int getWindowDim() const;
    float getEpsilonLow() const;
    float getEpsilonHigh() const;
    float getScalarBlue() const;
    float getScalarGreen() const;
    float getScalarRed() const;

private:
    // Some feasible default options.
    std::string suffix_ = "_decompressed";
    unsigned int tileDim_ = 5;
    unsigned int blockDim_ = 3;
    unsigned int windowDim_ = 10;
    float epsilonLow_ = 0;
    float epsilonHigh_ = 50.;
    float scalarBlue_ = 0.299f;
    float scalarGreen_ = 0.587f;
    float scalarRed_ = 0.114f;

    friend void GpuSimilarityFinder::setOptions(const SimilarityFinderOptions& options);
};

#pragma once

#include <string>
#include <utility>
#include <vector>

class SimilarityFinderOptions;

namespace GpuSimilarityFinder
{
    void setOptions(const SimilarityFinderOptions &options);
}

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

    SimilarityFinderOptions& setTileDim(unsigned int tileDim);
    SimilarityFinderOptions& setBlockDim(unsigned int blockDim);
    SimilarityFinderOptions& setWindowDim(unsigned int windowDim);
    SimilarityFinderOptions& setEpsilonLow(float epsilonLow);
    SimilarityFinderOptions& setEpsilonHigh(float epsilonHigh);
    SimilarityFinderOptions& setScalarBlue(float scalarBlue);
    SimilarityFinderOptions& setScalarGreen(float scalarGreen);
    SimilarityFinderOptions& setScalarRed(float scalarRed);

    unsigned int getTileDim() const;
    unsigned int getBlockDim() const;
    unsigned int getWindowDim() const;
    float getEpsilonLow() const;
    float getEpsilonHigh() const;
    float getScalarBlue() const;
    float getScalarGreen() const;
    float getScalarRed() const;

private:
    unsigned int tileDim_ = 5;
    unsigned int blockDim_ = 3;
    unsigned int windowDim_ = 10;
    float epsilonLow_ = 0.f;
    float epsilonHigh_ = 0.1f;
    float scalarBlue_ = 0.299f;
    float scalarGreen_ = 0.587f;
    float scalarRed_ = 0.114f;

    friend void GpuSimilarityFinder::setOptions(const SimilarityFinderOptions &options);
};

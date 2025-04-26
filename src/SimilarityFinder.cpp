#include <stdexcept>

#include "GpuSimilarityFinder.h"
#include "SimilarityFinder.h"

SimilarityFinder& SimilarityFinder::getInstance()
{
    static SimilarityFinder singleton;
    return singleton;
}

unsigned int SimilarityFinder::getPaddedSize(const unsigned int heightOrWidth) const
{
    if (!options_)
    {
        throw std::runtime_error("SimilarityFinder::getPaddedDimensions: no options were set before this call");
    }
    const unsigned int tileDim = options_->getTileDim();
    const unsigned int blockDim = options_->getBlockDim();

    const unsigned int paddedTiles = (heightOrWidth + tileDim - 1) / tileDim;
    const unsigned int paddedBlocks = ((paddedTiles + blockDim - 1) / blockDim);

    return paddedBlocks * blockDim * tileDim;
}

void SimilarityFinder::setOptions(const SimilarityFinderOptions &options)
{
    if (options.getEpsilonHigh() < options.getEpsilonLow())
    {
        throw std::runtime_error("SimilarityFinder::setOptions: the lower epsilon (" + std::to_string(options.getEpsilonLow()) +
                                 ") must not be greater than the higher epsilon (" + std::to_string(options.getEpsilonHigh()) + ")");
    }

    options_ = options;
    GpuSimilarityFinder::setOptions(options);
}

std::tuple<std::vector<unsigned int>, std::vector<float>, unsigned int, unsigned int> SimilarityFinder::findSimilarities(
    const unsigned int origHeight,
    const unsigned int origWidth,
    const uint8_t *const paddedImage,
    const unsigned int paddedHeight,
    const unsigned int paddedWidth) const
{
    if (!options_)
    {
        throw std::runtime_error("SimilarityFinder::findSimilarities: no options were set before this call");
    }
    const unsigned int tileDim = options_->getTileDim();
    const unsigned int blockDim = options_->getBlockDim();

    const auto [refsPadded, errorsPadded, numPaddedTilesWidth] =
        GpuSimilarityFinder::findSimilarities(tileDim, blockDim, paddedImage, paddedHeight, paddedWidth);

    const unsigned int numTilesHeight = (origHeight + tileDim - 1) / tileDim;
    const unsigned int numTilesWidth = (origWidth + tileDim - 1) / tileDim;
    const unsigned int numTiles = numTilesHeight * numTilesWidth;
        
    const unsigned int numBlocksHeight = (numTilesHeight + blockDim - 1) / blockDim;
    const unsigned int numBlocksWidth = (numTilesWidth + blockDim - 1) / blockDim;

    std::vector<unsigned int> refs(numTiles);
    std::vector<float> errors(numTiles);
    std::vector<bool> isReferenced(numTiles, false);

    for (unsigned int bh = numBlocksHeight; bh > 0; --bh)
    {
        for (unsigned int bw = numBlocksWidth; bw > 0; --bw)
        {
            for (unsigned int th = blockDim; th > 0; --th)
            {
                for (unsigned int tw = blockDim; tw > 0; --tw)
                {
                    const unsigned int tileIdxH = (bh - 1) * blockDim + (th - 1);
                    const unsigned int tileIdxW = (bw - 1) * blockDim + (tw - 1);
                    if (tileIdxH >= numTilesHeight || tileIdxW >= numTilesWidth)
                    {
                        continue;
                    }

                    const unsigned int paddedTileIdx = tileIdxH * numPaddedTilesWidth + tileIdxW;
                    const unsigned int tileIdx = tileIdxH * numTilesWidth + tileIdxW;

                    if (const unsigned int refPaddedTileIdx = refsPadded[paddedTileIdx];
                        refPaddedTileIdx == paddedTileIdx || isReferenced[tileIdx])
                    {
                        refs[tileIdx] = tileIdx;
                        errors[tileIdx] = 0;
                    }
                    else
                    {
                        const unsigned int refTileIdxH = refPaddedTileIdx / numPaddedTilesWidth;
                        const unsigned int refTileIdxW = refPaddedTileIdx % numPaddedTilesWidth;

                        if (refTileIdxH >= numTilesHeight || refTileIdxW >= numTilesWidth)
                        {
                            refs[tileIdx] = blackTileIdx;
                        }
                        else
                        {
                            const unsigned int refTileIdx = refTileIdxH * numTilesWidth + refTileIdxW;
                            refs[tileIdx] = refTileIdx;
                            isReferenced[refTileIdx] = true;
                        }
                        errors[tileIdx] = errorsPadded[paddedTileIdx];
                    }
                } 
            }
        }
    }

    return { std::move(refs), std::move(errors), numTilesHeight, numTilesWidth };
}

#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "GpuSimilarityFinder.h"

#define cudaCheck(call) \
  do { \
    const cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error at %s:%d: %s (%s)\n", __FILE__, __LINE__, cudaGetErrorString(err), #call); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

namespace {

__constant__ unsigned int d_tileDim;
__constant__ unsigned int d_blockDim;
__constant__ unsigned int d_windowDim;
__constant__ float d_epsilonLow;
__constant__ float d_epsilonHigh;
__constant__ float d_scalarBlue;
__constant__ float d_scalarGreen;
__constant__ float d_scalarRed;

__global__ void kernelSimFinder(
    unsigned int* const references,
    float* const errors,
    const uint8_t* const image,
    unsigned int* const referencesSliced,
    float* const errorsSliced,
    const unsigned int width,
    const unsigned int numTilesWidth)
{
    const unsigned int tileIdxH = blockIdx.x * d_blockDim + threadIdx.x;
    const unsigned int tileIdxW = blockIdx.y * d_blockDim + threadIdx.y;
    const unsigned int tileIdx = tileIdxH * numTilesWidth + tileIdxW;

    const unsigned int sliceIdxH = threadIdx.z / d_blockDim;
    const unsigned int sliceIdxW = threadIdx.z % d_blockDim;

    const unsigned int pixelIdxH = tileIdxH * d_tileDim;
    const unsigned int pixelIdxW = tileIdxW * d_tileDim;

    const float scalarAvg = 1. / (d_tileDim * d_tileDim);

    const float weightAvg = 0.5;
    float minError = 0;
    unsigned int bestRefTileIdx = tileIdx;

    for (unsigned int offsetH = min(d_windowDim, blockIdx.x + 1); offsetH > 0; --offsetH)
    {
        const unsigned int refBlockIdxH = blockIdx.x + 1 - offsetH;
        for (unsigned int offsetW = min(d_windowDim, blockIdx.y + 1); offsetW > 0; --offsetW)
        {
            const unsigned int refBlockIdxW = blockIdx.y + 1 - offsetW;
            
            if (refBlockIdxH == blockIdx.x && refBlockIdxW == blockIdx.y &&
                (sliceIdxH > threadIdx.x || sliceIdxH == threadIdx.x && sliceIdxW >= threadIdx.y))
            {
                continue;
            }

            const unsigned int refTileIdxH = refBlockIdxH * d_blockDim + sliceIdxH;
            const unsigned int refTileIdxW = refBlockIdxW * d_blockDim + sliceIdxW;
            const unsigned int refPixelIdxH = refTileIdxH * d_tileDim;
            const unsigned int refPixelIdxW = refTileIdxW * d_tileDim;

            float errorMax = 0;
            float errorAvg = 0;
            for (unsigned int x = 0; x < d_tileDim; ++x)
            {
                for (unsigned int y = 0; y < d_tileDim; ++y)
                {
                    const unsigned int idx = 3 * ((pixelIdxH + x) * width + (pixelIdxW + y));
                    const unsigned int refIdx = 3 * ((refPixelIdxH + x) * width + (refPixelIdxW + y));

                    // Relies on the fact that unsigned char is promoted to (signed) int before the subtractions.
                    const int diffBlue = image[idx] - image[refIdx];
                    const int diffGreen = image[idx + 1] - image[refIdx + 1];
                    const int diffRed = image[idx + 2] - image[refIdx + 2];

                    const float pixelError = d_scalarBlue * (diffBlue * diffBlue) + d_scalarGreen * (diffGreen * diffGreen) + d_scalarRed * (diffRed * diffRed);

                    errorAvg += pixelError;
                    errorMax = max(errorMax, pixelError);
                }
            }
            errorAvg *= scalarAvg;
            const float error = weightAvg * errorAvg + (1. - weightAvg) * errorMax;

            if (error <= d_epsilonHigh && (bestRefTileIdx == tileIdx || minError > d_epsilonLow && error < minError))
            {
                minError = error;
                bestRefTileIdx = refTileIdxH * numTilesWidth + refTileIdxW;
            }
        }
    }

    const unsigned int numSlices = d_blockDim * d_blockDim;
    const unsigned int resIdx = tileIdx * numSlices + threadIdx.z;
    referencesSliced[resIdx] = bestRefTileIdx;
    errorsSliced[resIdx] = minError;

    __syncthreads();

    if (threadIdx.z == 0)
    {
        for (size_t i = 1; i < numSlices; ++i)
        {
            const unsigned int refTileIdx = referencesSliced[resIdx + i];
            const float error = errorsSliced[resIdx + i];
            if (refTileIdx != tileIdx && error <= d_epsilonHigh &&
                (bestRefTileIdx == tileIdx ||
                 minError > d_epsilonLow && error < minError ||
                 error <= d_epsilonLow && refTileIdx < bestRefTileIdx))
            {
                bestRefTileIdx = refTileIdx;
                minError = error;
            }
        }
        references[tileIdx] = bestRefTileIdx;
        errors[tileIdx] = minError;
    }
}

} // anonymous namespace

namespace GpuSimilarityFinder {

void setOptions(const SimilarityFinderOptions& options)
{
    cudaCheck(cudaMemcpyToSymbol(d_tileDim, &options.tileDim_, sizeof(options.tileDim_)));
    cudaCheck(cudaMemcpyToSymbol(d_blockDim, &options.blockDim_, sizeof(options.blockDim_)));
    cudaCheck(cudaMemcpyToSymbol(d_windowDim, &options.windowDim_, sizeof(options.windowDim_)));
    cudaCheck(cudaMemcpyToSymbol(d_epsilonLow, &options.epsilonLow_, sizeof(options.epsilonLow_)));
    cudaCheck(cudaMemcpyToSymbol(d_epsilonHigh, &options.epsilonHigh_, sizeof(options.epsilonHigh_)));
    cudaCheck(cudaMemcpyToSymbol(d_scalarBlue, &options.scalarBlue_, sizeof(options.scalarBlue_)));
    cudaCheck(cudaMemcpyToSymbol(d_scalarGreen, &options.scalarGreen_, sizeof(options.scalarGreen_)));
    cudaCheck(cudaMemcpyToSymbol(d_scalarRed, &options.scalarRed_, sizeof(options.scalarRed_)));
}

std::tuple<std::vector<unsigned int>, std::vector<float>, unsigned int> findSimilarities(
    const unsigned int tileDim,
    const unsigned int blockDim,
    const uint8_t* const paddedImage,
    const unsigned int paddedHeight,
    const unsigned int paddedWidth)
{
    if (paddedHeight % tileDim != 0 || paddedWidth % tileDim != 0)
    {
        throw std::runtime_error("GpuSimilarityFinder::findSimilarities: input is not properly tile-padded");
    }
    const unsigned int numTilesHeight = paddedHeight / tileDim;
    const unsigned int numTilesWidth = paddedWidth / tileDim;
    
    if (numTilesHeight % blockDim != 0 || numTilesWidth % blockDim != 0)
    {
        throw std::runtime_error("GpuSimilarityFinder::findSimilarities: input is not properly block-padded");
    }
    const unsigned int numBlocksHeight = numTilesHeight / blockDim;
    const unsigned int numBlocksWidth = numTilesWidth / blockDim;
    
    const unsigned int numTiles = numTilesHeight * numTilesWidth;
    const unsigned int numSlices = blockDim * blockDim;

    // Allocate memory here so that any failure doesn't affect the Cuda related cleanup.
    std::vector<unsigned int> references(numTiles);
    std::vector<float> errors(numTiles);

    unsigned int* d_references;
    cudaCheck(cudaMalloc(&d_references, numTiles * sizeof(unsigned int)));
    float* d_errors;
    cudaCheck(cudaMalloc(&d_errors, numTiles * sizeof(float)));

    const unsigned int imageSize = paddedWidth * paddedHeight * 3 * sizeof(uint8_t);
    uint8_t* d_image;
    cudaCheck(cudaMalloc(&d_image, imageSize));
    cudaCheck(cudaMemcpy(d_image, paddedImage, imageSize, cudaMemcpyHostToDevice));
   
    const unsigned int numSlicedResElements = numTiles * numSlices;
    unsigned int* d_referencesSliced;
    cudaCheck(cudaMalloc(&d_referencesSliced, numSlicedResElements * sizeof(unsigned int)));
    float* d_errorsSliced;
    cudaCheck(cudaMalloc(&d_errorsSliced, numSlicedResElements * sizeof(float)));

    cudaEvent_t timerStart;
    cudaCheck(cudaEventCreate(&timerStart));
    cudaEvent_t timerEnd;
    cudaCheck(cudaEventCreate(&timerEnd));
    cudaCheck(cudaEventRecord(timerStart));

    const dim3 blocksPerGrid(numBlocksHeight, numBlocksWidth);
    const dim3 threadsPerBlock(blockDim, blockDim, numSlices);
    kernelSimFinder<<<blocksPerGrid, threadsPerBlock>>>(
        d_references, d_errors, d_image, d_referencesSliced, d_errorsSliced, paddedWidth, numTilesWidth);

    cudaCheck(cudaEventRecord(timerEnd));
    cudaCheck(cudaEventSynchronize(timerEnd));
    float kernelTimMs;
    cudaCheck(cudaEventElapsedTime(&kernelTimMs, timerStart, timerEnd));
    std::cout << "\nCUDA kernel execution time: " << kernelTimMs << " ms" << std::endl;

    cudaCheck(cudaMemcpy(references.data(), d_references, numTiles * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(errors.data(), d_errors, numTiles * sizeof(float), cudaMemcpyDeviceToHost));

	cudaCheck(cudaEventDestroy(timerStart));
    cudaCheck(cudaEventDestroy(timerEnd));

    cudaCheck(cudaFree(d_errorsSliced));
    cudaCheck(cudaFree(d_referencesSliced));
    cudaCheck(cudaFree(d_image));
    cudaCheck(cudaFree(d_errors));
    cudaCheck(cudaFree(d_references));

    return { std::move(references), std::move(errors), numTilesWidth };
}
}

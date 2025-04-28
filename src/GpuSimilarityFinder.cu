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

__device__ void loadBlockIntoSharedMemory(
    const uint8_t* const image,
    const unsigned int width,
    const unsigned int tileSize,
    const unsigned int blockIdxH,
    const unsigned int blockIdxW,
    uint8_t* const sharedData)
{
    const unsigned int tileFirstPixelIdxH = (blockIdxH * d_blockDim + threadIdx.x) * d_tileDim;
    const unsigned int tileFirstPixelIdxW = (blockIdxW * d_blockDim + threadIdx.y) * d_tileDim;
    const unsigned int tileFirstPixelIdx = tileFirstPixelIdxH * width + tileFirstPixelIdxW;

    const unsigned int tileFirstPixelIdxShared = (threadIdx.x * d_blockDim + threadIdx.y) * tileSize;

    for (unsigned int offset = threadIdx.z; offset < tileSize; offset += blockDim.z)
    {
        const unsigned int offsetH = offset / d_tileDim;
        const unsigned int offsetW = offset % d_tileDim;

        const unsigned int imageIdx = 3 * (tileFirstPixelIdx + offsetH * width + offsetW);
        const unsigned int sharedIdx = 3 * (tileFirstPixelIdxShared + offset);

        sharedData[sharedIdx] = image[imageIdx];
        sharedData[sharedIdx + 1] = image[imageIdx + 1];
        sharedData[sharedIdx + 2] = image[imageIdx + 2];
    }
}

__device__ float calcError(
    const uint8_t* const blockData1,
    const uint8_t* const blockData2,
    const unsigned int tileSize,
    const unsigned int tileFirstPixelIdxShared,
    const unsigned int refTileFirstPixelIdxShared)
{
    constexpr float weightAvg = 0.5;

    float errorMax = 0;
    float errorAvg = 0;
    for (unsigned int offset = 0; offset < tileSize; ++offset)
    {
        const unsigned int idx = 3 * (tileFirstPixelIdxShared + offset);
        const unsigned int refIdx = 3 * (refTileFirstPixelIdxShared + offset);

        // Relies on the fact that unsigned char is promoted to (signed) int before the subtractions.
        const int diffBlue = blockData1[idx] - blockData2[refIdx];
        const int diffGreen = blockData1[idx + 1] - blockData2[refIdx + 1];
        const int diffRed = blockData1[idx + 2] - blockData2[refIdx + 2];

        const float pixelError = d_scalarBlue * (diffBlue * diffBlue) + d_scalarGreen * (diffGreen * diffGreen) + d_scalarRed * (diffRed * diffRed);

        errorAvg += pixelError;
        errorMax = max(errorMax, pixelError);
    }
    errorAvg /= tileSize;
    
    return weightAvg * errorAvg + (1. - weightAvg) * errorMax;
}

__global__ void kernelSimFinder(
    unsigned int* const references,
    float* const errors,
    const uint8_t* const image,
    const unsigned int width,
    const unsigned int numTilesWidth)
{
    const unsigned int tileSize = d_tileDim * d_tileDim;
    const unsigned int numTilesPerBlock = d_blockDim * d_blockDim;
    const unsigned int numSlices = d_blockDim * d_blockDim;

    extern __shared__ float sharedData[];
    float* const errorsSliced = sharedData;
    static_assert(alignof(float) >= alignof(unsigned int));
    unsigned int* const refsSliced = (unsigned int*) (errorsSliced + numTilesPerBlock * numSlices);
    static_assert(alignof(unsigned int) >= alignof(uint8_t));
    uint8_t* const blockData1 = (uint8_t*) (refsSliced + numTilesPerBlock * numSlices);
    uint8_t* const blockData2 = blockData1 + 3 * numTilesPerBlock * tileSize;

    loadBlockIntoSharedMemory(image, width, tileSize, blockIdx.x, blockIdx.y, blockData1);

    __syncthreads();

    const unsigned int sliceIdxH = threadIdx.z / d_blockDim;
    const unsigned int sliceIdxW = threadIdx.z % d_blockDim;

    const unsigned int tileIdxBlock = threadIdx.x * d_blockDim + threadIdx.y;
    const unsigned int tileFirstPixelIdxShared = tileIdxBlock * tileSize;
    const unsigned int refTileFirstPixelIdxShared = (sliceIdxH * d_blockDim + sliceIdxW) * tileSize;

    const unsigned int tileIdx = (blockIdx.x * d_blockDim + threadIdx.x) * numTilesWidth + (blockIdx.y * d_blockDim + threadIdx.y);

    float minError = 0;
    unsigned int bestRefTileIdx = tileIdx;

    for (unsigned int offsetH = min(d_windowDim, blockIdx.x + 1); offsetH > 0; --offsetH)
    {
        const unsigned int refBlockIdxH = blockIdx.x + 1 - offsetH;
        for (unsigned int offsetW = min(d_windowDim, blockIdx.y + 1); offsetW > 0; --offsetW)
        {
            const unsigned int refBlockIdxW = blockIdx.y + 1 - offsetW;

            loadBlockIntoSharedMemory(image, width, tileSize, refBlockIdxH, refBlockIdxW, blockData2);

            __syncthreads();

            if (refBlockIdxH == blockIdx.x && refBlockIdxW == blockIdx.y &&
                (sliceIdxH > threadIdx.x || sliceIdxH == threadIdx.x && sliceIdxW >= threadIdx.y))
            {
                continue;
            }

            const float error = calcError(blockData1, blockData2, tileSize, tileFirstPixelIdxShared, refTileFirstPixelIdxShared);

            if (error <= d_epsilonHigh && (bestRefTileIdx == tileIdx || minError > d_epsilonLow && error < minError))
            {
                const unsigned int refTileIdx = (refBlockIdxH * d_blockDim + sliceIdxH) * numTilesWidth + (refBlockIdxW * d_blockDim + sliceIdxW);

                minError = error;
                bestRefTileIdx = refTileIdx;
            }
        }
    }

    const unsigned int resIdx = tileIdxBlock * numSlices + threadIdx.z;
    refsSliced[resIdx] = bestRefTileIdx;
    errorsSliced[resIdx] = minError;

    __syncthreads();

    if (threadIdx.z == 0)
    {
        for (size_t i = 1; i < numSlices; ++i)
        {
            const unsigned int refTileIdx = refsSliced[resIdx + i];
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
    
    const unsigned int numTilesTotal = numTilesHeight * numTilesWidth;

    // Allocate memory here so that any failure doesn't affect the Cuda related cleanup.
    std::vector<unsigned int> references(numTilesTotal);
    std::vector<float> errors(numTilesTotal);

    unsigned int* d_references;
    cudaCheck(cudaMalloc(&d_references, numTilesTotal * sizeof(unsigned int)));
    float* d_errors;
    cudaCheck(cudaMalloc(&d_errors, numTilesTotal * sizeof(float)));

    const unsigned int imageSize = paddedWidth * paddedHeight * 3 * sizeof(uint8_t);
    uint8_t* d_image;
    cudaCheck(cudaMalloc(&d_image, imageSize));
    cudaCheck(cudaMemcpy(d_image, paddedImage, imageSize, cudaMemcpyHostToDevice));
   
    cudaEvent_t timerStart;
    cudaCheck(cudaEventCreate(&timerStart));
    cudaEvent_t timerEnd;
    cudaCheck(cudaEventCreate(&timerEnd));
    cudaCheck(cudaEventRecord(timerStart));

    const unsigned int numTilesPerBlock = blockDim * blockDim;
    const unsigned int numSlices = blockDim * blockDim;

    const unsigned int blockDataSize = 3 * numTilesPerBlock * tileDim * tileDim * sizeof(uint8_t);
    const unsigned int scaledRefsSize = numTilesPerBlock * numSlices * sizeof(unsigned int);
    const unsigned int scaledErrorSize = numTilesPerBlock * numSlices * sizeof(float);
    const unsigned int sharedMemSize = 2 * blockDataSize + scaledRefsSize + scaledErrorSize;
    const dim3 blocksPerGrid(numBlocksHeight, numBlocksWidth);
    const dim3 threadsPerBlock(blockDim, blockDim, numSlices);
    kernelSimFinder<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_references, d_errors, d_image, paddedWidth, numTilesWidth);

    cudaCheck(cudaEventRecord(timerEnd));
    cudaCheck(cudaEventSynchronize(timerEnd));
    float kernelTimMs;
    cudaCheck(cudaEventElapsedTime(&kernelTimMs, timerStart, timerEnd));
    std::cout << "\nCUDA kernel execution time: " << kernelTimMs << " ms" << std::endl;

    cudaCheck(cudaMemcpy(references.data(), d_references, numTilesTotal * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(errors.data(), d_errors, numTilesTotal * sizeof(float), cudaMemcpyDeviceToHost));

	cudaCheck(cudaEventDestroy(timerStart));
    cudaCheck(cudaEventDestroy(timerEnd));

    cudaCheck(cudaFree(d_image));
    cudaCheck(cudaFree(d_errors));
    cudaCheck(cudaFree(d_references));

    return { std::move(references), std::move(errors), numTilesWidth };
}
}

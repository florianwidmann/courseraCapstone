#include <iostream>

#include "ImageIO.h"
#include "SimilarityFinder.h"

namespace
{
    
    std::vector<uint8_t> createOutputImage(
        const std::vector<uint8_t>& paddedImage,
        const unsigned int origHeight,
        const unsigned int origWidth,
        const unsigned int paddedWidth,
        const std::vector<unsigned int>& refs,
        const unsigned int numTilesHeight,
        const unsigned int numTilesWidth,
        const unsigned int tileDim)
    {
        std::vector<uint8_t> resImage(origHeight * origWidth * 3);
        for (unsigned int tileIdx = 0; tileIdx < refs.size(); ++tileIdx)
        {
            const unsigned int tileIdxH = tileIdx / numTilesWidth;
            const unsigned int tileIdxW = tileIdx % numTilesWidth;

            const unsigned int refTileIdx = refs[tileIdx];
            const bool isBlackTile = refTileIdx == SimilarityFinder::blackTileIdx;
            const unsigned int refTileIdxH = refTileIdx / numTilesWidth;
            const unsigned int refTileIdxW = refTileIdx % numTilesWidth;

            for (unsigned int x = 0; x < tileDim; ++x)
            {
                const unsigned int pixelIdxH = tileIdxH * tileDim + x;
                if (pixelIdxH >= origHeight)
                {
                    break;
                }
                const unsigned int refPixelIdxH = refTileIdxH * tileDim + x;

                for (unsigned int y = 0; y < tileDim; ++y)
                {
                    const unsigned int pixelIdxW = tileIdxW * tileDim + y;
                    if (pixelIdxW >= origWidth)
                    {
                        break;
                    }
                    const unsigned int refPixelIdxW = refTileIdxW * tileDim + y;

                    const unsigned int idx = 3 * (pixelIdxH * origWidth + pixelIdxW);
                    const unsigned int refIdx = 3 * (refPixelIdxH * paddedWidth + refPixelIdxW);

                    resImage[idx] = isBlackTile ? 0 : paddedImage[refIdx];
                    resImage[idx + 1] = isBlackTile ? 0 : paddedImage[refIdx + 1];
                    resImage[idx + 2] = isBlackTile ? 0 : paddedImage[refIdx + 2];
                }
            }
        }

        return resImage;
    }

    // TODO: Could be made more flexible.
    std::string getOutputFileName(const std::string &inputFile)
    {
        const std::string preExtSuffix = "_compressed";
        const std::string::size_type dotLoc = inputFile.rfind('.');
        return dotLoc == std::string::npos ? inputFile + preExtSuffix : inputFile.substr(0, dotLoc) + preExtSuffix + inputFile.substr(dotLoc);
    }
}

int main(int argc, char *argv[])
{
    try
    {
        const auto [inputFiles, options] = SimilarityFinderOptions::parseFromArgLine(argc, argv);

        if (inputFiles.empty())
        {
            std::cout << "no files specified and hence none converted" << std::endl;
            return 0;
        }

        SimilarityFinder &finder = SimilarityFinder::getInstance();
        finder.setOptions(options);
        const unsigned int tileDim = options.getTileDim();

        for (const std::string &inputFile : inputFiles)
        {
            try
            {
                std::cout << "\nprocessing file " << inputFile << "...";

                const auto [format, paddedImage, origHeight, origWidth, paddedHeight, paddedWidth] =
                    ImageIO::loadPaddedImage(inputFile, [&](const unsigned int heightOrWidth) { return finder.getPaddedSize(heightOrWidth); });

                const auto [refs, errors, numTilesHeight, numTilesWidth] =
                    finder.findSimilarities(origHeight, origWidth, paddedImage.data(), paddedHeight, paddedWidth);

                const std::vector<uint8_t> outputImage =
                    createOutputImage(paddedImage, origHeight, origWidth, paddedWidth, refs, numTilesHeight, numTilesWidth, tileDim);

                const std::string outputFile = getOutputFileName(inputFile);
                ImageIO::saveImage(outputFile, outputImage, origHeight, origWidth, origWidth, format);

                std::cout << " done" << std::endl;

                size_t numRefTiles = 0;
                float maxError = 0;
                float avgError = 0;
                for (size_t i = 0; i < refs.size(); ++i)
                {
                    if (refs[i] != i)
                    {
                        ++numRefTiles;
                        maxError = std::max(maxError, errors[i]);
                        avgError += errors[i];
                    }
                }
                const float percentage = static_cast<float>(numRefTiles) / refs.size() * 100.;
                avgError = numRefTiles == 0 ? 0 : avgError / numRefTiles;

                std::cout << "total number of tiles: " << refs.size() << "\n";
                std::cout << "number of reference tiles: " << numRefTiles << " (" << percentage << "%)\n";
                std::cout << "max error: " << maxError << "\n";
                std::cout << "average error (only considering reference tiles): " << avgError << "\n";
                std::cout << std::endl;
            }
            catch (const std::exception &exn)
            {
                std::cerr << "\nProgram error! The following exception occurred for input file " << inputFile << ": \n";
                std::cerr << exn.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "\nProgram error! An unknow type of exception occurred for input file " << inputFile << std::endl;
            }
        }
    }
    catch (const std::exception &exn)
    {
        std::cerr << "\nProgram error! The following exception occurred:\n";
        std::cerr << exn.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "\nProgram error! An unknow type of exception occurred." << std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

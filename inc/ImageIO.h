#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <functional>

#include <FreeImage.h>

namespace ImageIO
{
    std::tuple<FREE_IMAGE_FORMAT, std::vector<uint8_t>, unsigned int, unsigned int, unsigned int, unsigned int> loadPaddedImage(
        const std::string &inputFile,
        const std::function<unsigned int(unsigned int)> &paddingFct);

    void saveImage(
        const std::string &outputFile,
        const std::vector<uint8_t> &outputImage,
        unsigned int height,
        unsigned int width,
        unsigned int paddedWidth,
        FREE_IMAGE_FORMAT format);
}

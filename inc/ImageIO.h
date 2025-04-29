#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <functional>

#include <FreeImage.h>

namespace ImageIO
{
    /**
     * Loads, converts and padds an image.
     * \param[in] inputFile The image file path.
     * \param[in] paddingFct Given a height or width, returns the padded height or width, respectively.
     * \return
     * 1) The image format.
     * 2) The bitmap in row-based, left to right, top-down order. Every pixel is represented as a blue, green and res value.
     *    Black padding is applied for both height (at the bottom) and width (at the right) as determined by the padding parameter.
     * 3) The original height.
     * 4) The original width.
     * 5) The padded height.
     * 6) The padded width.
     */
    std::tuple<FREE_IMAGE_FORMAT, std::vector<uint8_t>, unsigned int, unsigned int, unsigned int, unsigned int> loadPaddedImage(
        const std::string& inputFile,
        const std::function<unsigned int(unsigned int)>& paddingFct);

    /**
     * Saves an image.
     * \param[in] outputFile The file path of the output image.
     * \param[in] outputImage The bitmap in row-based, left to right, top-down order. Every pixel is represented as a blue, green and res value.
     * The image might be padded.
     * \param[in] height The image height.
     * \param[in] width The image width.
     * \param[in] paddedWidth The padded width.
     * \param[in] format The requested image format.
     */
    void saveImage(
        const std::string& outputFile,
        std::vector<uint8_t>&& outputImage,
        unsigned int height,
        unsigned int width,
        unsigned int paddedWidth,
        FREE_IMAGE_FORMAT format);
}

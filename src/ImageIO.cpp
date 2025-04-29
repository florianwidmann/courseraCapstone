#include <cstring>

#include "ImageIO.h"

namespace ImageIO
{
    std::tuple<FREE_IMAGE_FORMAT, std::vector<uint8_t>, unsigned int, unsigned int, unsigned int, unsigned int> loadPaddedImage(
        const std::string& inputFile,
        const std::function<unsigned int(unsigned int)>& paddingFct)
    {
        FREE_IMAGE_FORMAT format = FreeImage_GetFileType(inputFile.c_str());
        if (format == FIF_UNKNOWN)
        {
            // If the image signature cannot be determined (some file formats don't have any), try using the file extension.
            format = FreeImage_GetFIFFromFilename(inputFile.c_str());
        }
        if (format == FIF_UNKNOWN)
        {
            throw std::runtime_error("unknown free image format for file " + inputFile);
        }
        if (!FreeImage_FIFSupportsReading(format))
        {
            throw std::runtime_error("unsupported free image format " + std::to_string(format) + " for file " + inputFile);
        }

        FIBITMAP* const origBitmap = FreeImage_Load(format, inputFile.c_str());
        if (!origBitmap)
        {
            throw std::runtime_error("error loading (free) image for file " + inputFile);
        }

        FIBITMAP* const convBitmap = FreeImage_ConvertTo24Bits(origBitmap);
        if (!convBitmap)
        {
            throw std::runtime_error("error converting (free) image for file " + inputFile);
        }

        // TODO: This could be made more flexible.
        unsigned int redMask = FreeImage_GetRedMask(convBitmap);
        unsigned int greenMask = FreeImage_GetGreenMask(convBitmap);
        unsigned int blueMask = FreeImage_GetBlueMask(convBitmap);
        if (redMask != 0x00FF0000 || greenMask != 0x0000FF00 || blueMask != 0x000000FF)
        {
            throw std::runtime_error("unsupported RGB mask for file " + inputFile);
        }

        const unsigned int origHeight = FreeImage_GetHeight(convBitmap);
        const unsigned int origWidth = FreeImage_GetWidth(convBitmap);
        if (origHeight == 0 || origWidth == 0)
        {
            throw std::runtime_error("input image has zero dimensions for file " + inputFile);
        }

        const unsigned int paddedHeight = paddingFct(origHeight);
        const unsigned int paddedWidth = paddingFct(origWidth);

        std::vector<uint8_t> image(paddedHeight * paddedWidth * 3, 0);

        // We cannot use FreeImage_ConvertToRawBits here since it doesn't support the change in pitch.
        const unsigned int srcPitch = FreeImage_GetPitch(convBitmap);
        const uint8_t* srcLine = FreeImage_GetBits(convBitmap) + srcPitch * (origHeight - 1);
        uint8_t* dstLine = image.data();
        const unsigned int dstPitch = paddedWidth * 3 * sizeof(uint8_t);

        for (size_t line = 0; line < origHeight; ++line)
        {
            memcpy(dstLine, srcLine, origWidth * 3 * sizeof(uint8_t));
            srcLine -= srcPitch;
            dstLine += dstPitch;
        }

        FreeImage_Unload(origBitmap);
        FreeImage_Unload(convBitmap);

        return {format, std::move(image), origHeight, origWidth, paddedHeight, paddedWidth};
    }

    void saveImage(
        const std::string& outputFile,
        std::vector<uint8_t>&& outputImage,
        const unsigned int height,
        const unsigned int width,
        const unsigned int paddedWidth,
        const FREE_IMAGE_FORMAT format)
    {
        FIBITMAP* const resultBitmap = FreeImage_ConvertFromRawBits(outputImage.data(), width, height, 3 * paddedWidth, 24, 0x00FF0000, 0x0000FF00, 0x000000FF, TRUE);

        const bool success = FreeImage_Save(format, resultBitmap, outputFile.c_str(), 0) == TRUE;
        if (!success)
        {
            throw std::runtime_error("error saving image for file " + outputFile);
        }

        FreeImage_Unload(resultBitmap);
    }
}

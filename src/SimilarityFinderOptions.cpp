#include <iostream>
#include <stdexcept>

#include "SimilarityFinderOptions.h"

namespace {

unsigned int readUnsignedIntArg(int& i, const int argc, char *argv[], const std::string& arg, const int maxVal)
{
    ++i;
    if (i >= argc)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has no argument");
    }

    int val;
    try
    {
        // TODO: Could disallow "garbage" at the end and check for "too big" values.
        val = std::stol(argv[i]);
    } catch (...)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has bad argument: " + argv[i]);
    }

    if (val < 0 || val > maxVal)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has value outside the allowed range [0, " + std::to_string(maxVal) + "]: " + argv[i]);
    }

    return val;
}

float readFloatArg(int& i, const int argc, char *argv[], const std::string& arg)
{
    ++i;
    if (i >= argc)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has no argument");
    }

    try
    {
        // TODO: Could disallow "garbage" at the end and check for "too big" values.
        return std::stof(argv[i]);
    }
    catch (...)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has bad argument: " + argv[i]);
    }
}

std::string readStringArg(int &i, const int argc, char *argv[], const std::string &arg)
{
    ++i;
    if (i >= argc)
    {
        throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: flag '" + arg + "' has no argument");
    }

    return argv[i];
}

void printUsage()
{
    std::cout << "purpose: simulates compressing image files using similarities of 'tiles' in the image\n";
    std::cout << "usage: <executable> <flags> <file names>, where\n";
    std::cout << "<file names>: zero or more image files\n";
    std::cout << "<flags>: zero or more of the following flags with a value (as next argument) each\n";
    std::cout << "--suffix: this suffix is added to the original file name (right before the extension, if any) to get the decompressed image file name\n";
    std::cout << "--tileDim: the dimension (both height and width) of a tile in pixels\n";
    std::cout << "--blockDim: the dimension (both height and width) of an internal block in tiles\n";
    std::cout << "--windowDim: the dimension (both height and width) of the sliding window in blocks\n";
    std::cout << "--epsLow: the lower epsilon (meaningful range [0, 65,025])\n";
    std::cout << "--epsHigh: the higher epsilon (meaningful range [0, 65,025])\n";
    std::cout << "--scalarBlue: the scalar for the difference calculation of the blue channel\n";
    std::cout << "--scalarGreen: the scalar for the difference calculation of the green channel\n";
    std::cout << "--scalarRed: the scalar for the difference calculation of the red channel\n";
    exit(0);
}

} // anonymous namespace

// TODO: This could be made more sophisticated, e.g. use a professional argument parsing library.
std::pair<std::vector<std::string>, SimilarityFinderOptions> SimilarityFinderOptions::parseFromArgLine(
    const int argc,
    char *argv[])
{
    std::vector<std::string> inputFiles;
    SimilarityFinderOptions options;

    // slight overapproximation
    inputFiles.reserve(argc);

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg.find("--") == 0)
        {
            const std::string flag = arg.substr(2);
            if (flag == "suffix")
            {
                options.setSuffix(readStringArg(i, argc, argv, arg));
            }
            if (flag == "tileDim")
            {
                options.setTileDim(readUnsignedIntArg(i, argc, argv, arg, 20));
            }
            else if (flag == "blockDim")
            {
                options.setBlockDim(readUnsignedIntArg(i, argc, argv, arg, 10));
            }
            else if (flag == "windowDim")
            {
                options.setWindowDim(readUnsignedIntArg(i, argc, argv, arg, 1000));
            }
            else if (flag == "epsLow")
            {
                options.setEpsilonLow(readFloatArg(i, argc, argv, arg));
            }
            else if (flag == "epsHigh")
            {
                options.setEpsilonHigh(readFloatArg(i, argc, argv, arg));
            }
            else if (flag == "scalarBlue")
            {
                options.setScalarBlue(readFloatArg(i, argc, argv, arg));
            }
            else if (flag == "scalarGreen")
            {
                options.setScalarGreen(readFloatArg(i, argc, argv, arg));
            }
            else if (flag == "scalarRed")
            {
                options.setScalarRed(readFloatArg(i, argc, argv, arg));
            }
            else if (flag == "help")
            {
                printUsage();
            }
            else
            {
                throw std::runtime_error("SimilarityFinderOptions::parseFromArgLine: unsupported flag '" + arg + "'");
            }
        }
        else
        {
            inputFiles.push_back(std::move(arg));
        }
    }

    return { std::move(inputFiles), std::move(options) };
}

SimilarityFinderOptions& SimilarityFinderOptions::setSuffix(const std::string& suffix)
{
    if (suffix.empty())
    {
        throw std::runtime_error("SimilarityFinderOptions::setSuffix: suffix must not be empty");
    }

    suffix_ = suffix;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setTileDim(const unsigned int tileDim)
{
    if (tileDim == 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setTileDim: tile dimension must be positive");
    }

    tileDim_ = tileDim;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setBlockDim(const unsigned int blockDim)
{
    if (blockDim == 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setBlockDim: block dimension must be positive");
    }

    blockDim_ = blockDim;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setWindowDim(const unsigned int windowDim)
{
    if (windowDim == 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setWindowDim: window dimension must be positive");
    }

    windowDim_ = windowDim;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setEpsilonLow(const float epsilonLow)
{
    if (epsilonLow < 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setEpsilonLow: lower epsilon must be non-negative");
    }

    epsilonLow_ = epsilonLow;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setEpsilonHigh(const float epsilonHigh)
{
    if (epsilonHigh < 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setEpsilonHigh: higher epsilon must be non-negative");
    }

    epsilonHigh_ = epsilonHigh;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setScalarBlue(const float scalarBlue)
{
    if (scalarBlue <= 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setScalarBlue: blue scalar must be positive");
    }

    scalarBlue_ = scalarBlue;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setScalarGreen(const float scalarGreen)
{
    if (scalarGreen <= 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setScalarGreen: green scalar must be positive");
    }

    scalarGreen_ = scalarGreen;
    return *this;
}

SimilarityFinderOptions& SimilarityFinderOptions::setScalarRed(const float scalarRed)
{
    if (scalarRed <= 0)
    {
        throw std::runtime_error("SimilarityFinderOptions::setScalarRed: red scalar must be positive");
    }

    scalarRed_ = scalarRed;
    return *this;
}

const std::string& SimilarityFinderOptions::getSuffix() const
{
    return suffix_;
}

unsigned int SimilarityFinderOptions::getTileDim() const
{
    return tileDim_;
}

unsigned int SimilarityFinderOptions::getBlockDim() const
{
    return blockDim_;
}

unsigned int SimilarityFinderOptions::getWindowDim() const
{
    return windowDim_;
}

float SimilarityFinderOptions::getEpsilonLow() const
{
    return epsilonLow_;
}

float SimilarityFinderOptions::getEpsilonHigh() const
{
    return epsilonHigh_;
}

float SimilarityFinderOptions::getScalarBlue() const
{
    return scalarBlue_;
}

float SimilarityFinderOptions::getScalarGreen() const
{
    return scalarGreen_;
}

float SimilarityFinderOptions::getScalarRed() const
{
    return scalarRed_;
}

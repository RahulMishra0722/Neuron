#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

/**
 * Reads a 32-bit integer from binary file in big-endian format (MNIST standard)
 * @param file Input file stream
 * @return 32-bit integer in host byte order
 */
uint32_t readBigEndianInt32(std::ifstream &file)
{
  uint32_t value;
  file.read(reinterpret_cast<char *>(&value), 4);
  return __builtin_bswap32(value); // Convert from big-endian to host byte order
}

/**
 * Loads MNIST image data from binary file
 * @param filename Path to MNIST image file (e.g., "train-images-idx3-ubyte")
 * @param images Output vector to store normalized image data
 * @return true if successful, false otherwise
 */
bool loadMNISTImages(const string &filename, vector<vector<double>> &images)
{
  std::ifstream imageFile(filename, std::ios::binary);
  if (!imageFile)
  {
    cerr << "Failed to open image file: " << filename << endl;
    return false;
  }

  // Read header information
  uint32_t magicNumber = readBigEndianInt32(imageFile);
  uint32_t numImages = readBigEndianInt32(imageFile);
  uint32_t numRows = readBigEndianInt32(imageFile);
  uint32_t numCols = readBigEndianInt32(imageFile);

  cout << "Loading " << filename << ":" << endl;
  cout << "  Magic number: " << magicNumber << endl;
  cout << "  Number of images: " << numImages << endl;
  cout << "  Image dimensions: " << numRows << " x " << numCols << endl;

  // Verify magic number for image files
  if (magicNumber != 2051)
  {
    cerr << "Invalid magic number for image file: " << magicNumber << " (expected 2051)" << endl;
    return false;
  }

  int pixelsPerImage = numRows * numCols;
  cout << "  Pixels per image: " << pixelsPerImage << endl;

  // Reserve space for efficiency
  images.clear();
  images.reserve(numImages);

  // Load each image
  for (uint32_t imageIndex = 0; imageIndex < numImages; imageIndex++)
  {
    vector<double> currentImage;
    currentImage.reserve(pixelsPerImage);

    // Read each pixel
    for (int pixelIndex = 0; pixelIndex < pixelsPerImage; pixelIndex++)
    {
      uint8_t rawPixelValue;
      imageFile.read(reinterpret_cast<char *>(&rawPixelValue), sizeof(uint8_t));

      // Check for premature end of file
      if (imageFile.eof())
      {
        cerr << "Unexpected end of file at image " << imageIndex
             << ", pixel " << pixelIndex << endl;
        return false;
      }

      // Normalize pixel value from [0-255] to [0-1]
      double normalizedPixel = static_cast<double>(rawPixelValue) / 255.0;
      currentImage.push_back(normalizedPixel);
    }

    // Progress indicator
    if (imageIndex % 5000 == 0)
    {
      cout << "  Loaded " << imageIndex << " images..." << endl;
    }

    images.push_back(std::move(currentImage));
  }

  imageFile.close();
  cout << "  Successfully loaded " << images.size() << " images" << endl;
  return true;
}

/**
 * Loads MNIST label data from binary file
 * @param filename Path to MNIST label file (e.g., "train-labels-idx1-ubyte")
 * @param labels Output vector to store label data
 * @return true if successful, false otherwise
 */
bool loadMNISTLabels(const string &filename, vector<int> &labels)
{
  std::ifstream labelFile(filename, std::ios::binary);
  if (!labelFile)
  {
    cerr << "Failed to open label file: " << filename << endl;
    return false;
  }

  // Read header information
  uint32_t magicNumber = readBigEndianInt32(labelFile);
  uint32_t numLabels = readBigEndianInt32(labelFile);

  cout << "Loading " << filename << ":" << endl;
  cout << "  Magic number: " << magicNumber << endl;
  cout << "  Number of labels: " << numLabels << endl;

  // Verify magic number for label files
  if (magicNumber != 2049)
  {
    cerr << "Invalid magic number for label file: " << magicNumber << " (expected 2049)" << endl;
    return false;
  }

  // Reserve space for efficiency
  labels.clear();
  labels.reserve(numLabels);

  // Load each label
  for (uint32_t labelIndex = 0; labelIndex < numLabels; labelIndex++)
  {
    uint8_t rawLabel;
    labelFile.read(reinterpret_cast<char *>(&rawLabel), sizeof(uint8_t));

    // Check for premature end of file
    if (labelFile.eof())
    {
      cerr << "Unexpected end of file at label " << labelIndex << endl;
      return false;
    }

    labels.push_back(static_cast<int>(rawLabel));
  }

  labelFile.close();
  cout << "  Successfully loaded " << labels.size() << " labels" << endl;
  return true;
}

/**
 * Validates that image and label data are consistent
 * @param images Vector of image data
 * @param labels Vector of label data
 * @param expectedPixelsPerImage Expected number of pixels per image (784 for MNIST)
 * @return true if data is valid, false otherwise
 */
bool validateData(const vector<vector<double>> &images, const vector<int> &labels, int expectedPixelsPerImage)
{
  if (images.size() != labels.size())
  {
    cerr << "Data validation failed: " << images.size()
         << " images but " << labels.size() << " labels" << endl;
    return false;
  }

  if (images.empty())
  {
    cerr << "Data validation failed: No images loaded" << endl;
    return false;
  }

  if (images[0].size() != expectedPixelsPerImage)
  {
    cerr << "Data validation failed: Expected " << expectedPixelsPerImage
         << " pixels per image, got " << images[0].size() << endl;
    return false;
  }

  cout << "Data validation successful:" << endl;
  cout << "  " << images.size() << " images and labels loaded" << endl;
  cout << "  " << images[0].size() << " pixels per image" << endl;

  cout << "  Sample data - First image label: " << labels[0] << endl;
  cout << "  First 10 pixels: ";
  for (int i = 0; i < min(10, static_cast<int>(images[0].size())); i++)
  {
    cout << images[0][i] << " ";
  }
  cout << endl;

  return true;
}

int main()
{
  NeuralNetwork neuralNet;
  neuralNet.addLayer(784, 256); // Input layer: 784 -> 256
  neuralNet.addLayer(256, 128); // Hidden layer: 256 -> 128
  neuralNet.addLayer(128, 64);  // Hidden layer: 128 -> 64
  neuralNet.addLayer(64, 10);   // Output layer: 64 -> 10

  vector<vector<double>> trainingImages;
  vector<int> trainingLabels;
  vector<vector<double>> testingImages;
  vector<int> testingLabels;

  cout << "=== Loading MNIST Training Data ===" << endl;

  if (!loadMNISTImages("train-images-idx3-ubyte", trainingImages))
  {
    cerr << "Failed to load training images" << endl;
    return -1;
  }

  if (!loadMNISTLabels("train-labels-idx1-ubyte", trainingLabels))
  {
    cerr << "Failed to load training labels" << endl;
    return -1;
  }

  if (!validateData(trainingImages, trainingLabels, 784))
  {
    cerr << "Training data validation failed" << endl;
    return -1;
  }

  cout << "\n=== Loading MNIST Testing Data ===" << endl;

  if (!loadMNISTImages("t10k-images-idx3-ubyte", testingImages))
  {
    cerr << "Failed to load testing images" << endl;
    return -1;
  }

  if (!loadMNISTLabels("t10k-labels-idx1-ubyte", testingLabels))
  {
    cerr << "Failed to load testing labels" << endl;
    return -1;
  }

  if (!validateData(testingImages, testingLabels, 784))
  {
    cerr << "Testing data validation failed" << endl;
    return -1;
  }

  cout << "\n=== Starting Neural Network Training ===" << endl;
  neuralNet.train(trainingImages, trainingLabels, 30);

  cout << "\n=== Testing Neural Network ===" << endl;
  neuralNet.testModel(testingImages, testingLabels);

  cout << "\n=== Training and Testing Complete ===" << endl;
  return 0;
}

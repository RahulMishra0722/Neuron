#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

uint32_t readInt32(std::ifstream &file)
{
  uint32_t value;
  file.read(reinterpret_cast<char *>(&value), 4);
  return __builtin_bswap32(value);
}

int main()
{
  NeuralNetwork nn;
  nn.addLayer(784, 256); // Hidden layer: 784 -> 128
  nn.addLayer(256, 128); // Hidden layer: 784 -> 128
  nn.addLayer(128, 64);  // Output layer: 128 -> 10
  nn.addLayer(64, 10);   // Output layer: 128 -> 10

  vector<int> labels;
  vector<vector<double>> training_data;
  vector<vector<double>> inferenceData;
  vector<vector<double>> inferenceLabels;

  // Load labels
  ifstream labelsFile("train-labels-idx1-ubyte", ios::binary);
  if (!labelsFile)
  {
    cerr << "Failed to open labels file.\n";
    return -1;
  }

  uint32_t lmagic = readInt32(labelsFile);
  uint32_t numOfLabels = readInt32(labelsFile);

  cout << "Labels file magic number: " << lmagic << ", Labels: " << numOfLabels << endl;

  for (int label_idx = 0; label_idx < numOfLabels; label_idx++)
  {
    uint8_t rawByte;
    labelsFile.read(reinterpret_cast<char *>(&rawByte), sizeof(uint8_t));

    if (label_idx < 10)
    {
      cout << "Label " << label_idx << ": " << (int)rawByte << endl;
    }

    labels.push_back(rawByte);
  }
  labelsFile.close();

  // Load images
  std::ifstream mnist_training_data_file("train-images-idx3-ubyte", std::ios::binary);
  if (!mnist_training_data_file)
  {
    cerr << "Failed to open image file.\n";
    return -1;
  }

  uint32_t magic = readInt32(mnist_training_data_file);
  uint32_t numImages = readInt32(mnist_training_data_file);
  uint32_t rows = readInt32(mnist_training_data_file);
  uint32_t cols = readInt32(mnist_training_data_file);

  cout << "Image file magic number: " << magic << ", Images: " << numImages
       << ", Rows: " << rows << ", Cols: " << cols << endl;

  // Verify magic numbers
  if (magic != 2051)
  {
    cerr << "Invalid magic number for images file: " << magic << endl;
    return -1;
  }

  int ImageSize = rows * cols;
  cout << "Image size: " << ImageSize << " pixels" << endl;

  // Reserve space for efficiency
  training_data.reserve(numImages);

  for (int image = 0; image < numImages; image++)
  {
    vector<double> currentImage;
    currentImage.reserve(ImageSize); // Reserve space for efficiency

    for (int pixel = 0; pixel < ImageSize; pixel++)
    {
      uint8_t rawByte;
      mnist_training_data_file.read(reinterpret_cast<char *>(&rawByte), sizeof(uint8_t));

      if (mnist_training_data_file.eof())
      {
        cerr << "Unexpected end of file at image " << image << ", pixel " << pixel << endl;
        return -1;
      }

      double normalized = static_cast<double>(rawByte) / 255.0;
      currentImage.push_back(normalized);
    }

    if (image % 1000 == 0)
    {
      cout << "Loaded " << image << " images" << endl;
    }

    training_data.push_back(std::move(currentImage)); // Use move for efficiency
  }
  mnist_training_data_file.close();

  cout << "Loaded " << training_data.size() << " images and " << labels.size() << " labels" << endl;

  if (training_data.size() != labels.size())
  {
    cerr << "Mismatch: " << training_data.size() << " images but " << labels.size() << " labels" << endl;
    return -1;
  }

  if (!training_data.empty())
  {
    cout << "First image has " << training_data[0].size() << " pixels" << endl;
    cout << "Expected: " << ImageSize << " pixels" << endl;

    cout << "First few pixels of first image: ";
    for (int i = 0; i < min(10, (int)training_data[0].size()); i++)
    {
      cout << training_data[0][i] << " ";
    }
    cout << endl;
    cout << "Label for first image: " << labels[0] << endl;
  }

  cout << "Starting training..." << endl;
  nn.train(training_data, labels, 30);

  return 0;
}

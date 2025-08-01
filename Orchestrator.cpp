#include "neuron_node.h" // Changed from <neuron.h> to "neuron.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
using namespace std;

uint32_t readInt32(std::ifstream &file)
{
  uint32_t value;
  file.read(reinterpret_cast<char *>(&value), 4);
  return __builtin_bswap32(value);
}

class Orchestrator
{
  NeuronNode n;

public:
  double loadDataAndTrainTest()
  {
    vector<double> labels;
    std::ifstream label_file("train-labels-idx1-ubyte", std::ios::binary);
    if (!label_file)
    {
      cerr << "Failed to open label file.\n";
      return -1;
    }

    uint32_t lmagic = readInt32(label_file);
    uint32_t numOfLabels = readInt32(label_file);
    cout << "Label file magic number: " << lmagic << ", Number of labels: " << numOfLabels << endl;

    for (uint32_t label = 0; label < numOfLabels; label++)
    {
      uint8_t rawLabel;
      label_file.read(reinterpret_cast<char *>(&rawLabel), sizeof(uint8_t));
      labels.push_back(rawLabel);
    }

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

    int ImageSize = rows * cols;
    vector<double> flattedArray;

    for (uint32_t image = 0; image < numImages; image++)
    {
      if (image % 1000 == 0)
      {
        cout << "Processing image " << image << " / " << numImages << " ("
             << (100.0 * image / numImages) << "% done)\n";
      }

      for (int pixel = 0; pixel < ImageSize; pixel++)
      {
        uint8_t rawByte;
        mnist_training_data_file.read(reinterpret_cast<char *>(&rawByte), sizeof(uint8_t));
        double normalized = rawByte / 255.0;
        flattedArray.push_back(normalized);
      }

      n.forward_pass(flattedArray);
      double binary_target = (labels[image] == 5) ? 1.0 : 0.0;
      int output = n.backward_pass(labels[image]);

      cout << "Image #" << image << ": Predicted = " << output
           << ", Actual = " << labels[image] << endl;

      flattedArray.clear();
    }

    cout << "Training complete.\n";
    return 0;
  };
};

int main()
{
  Orchestrator o;
  o.loadDataAndTrainTest();
}

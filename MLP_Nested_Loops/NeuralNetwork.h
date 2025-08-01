#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <iostream>
#include "Layer.h"

using namespace std;

class NeuralNetwork
{
private:
  vector<Layer> layers;
  vector<double> last_network_output;

  // Helper functions
  vector<double> createOneHotVector(int label, int num_classes);
  int getPredictedClass(vector<double> &output);
  void printConfusionMatrix(vector<vector<int>> &matrix);

public:
  NeuralNetwork();

  void addLayer(int input_size, int output_size);
  void train(vector<vector<double>> &training_data, vector<int> &labels, int epochs);

  vector<double> forward_pass(vector<double> input);
  vector<double> backward_pass(vector<double> target);

  double calculateLoss(vector<double> predicted, vector<double> targets);
  double calculateMSELoss(vector<double> predicted, vector<double> targets);

  // Validation and testing functions
  double validateModel(vector<vector<double>> &data, vector<int> &labels,
                       int training_size, int validation_size);
  double testModel(vector<vector<double>> &test_data, vector<int> &test_labels);
};

#endif

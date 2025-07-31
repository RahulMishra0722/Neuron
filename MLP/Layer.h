#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include "Neuron.h"
using namespace std;
class Layer
{
private:
  vector<Neuron> neurons;
  int input_size;
  int output_size;

public:
  Layer(int input_size, int output_size);
  vector<double> forward_pass(vector<double> input);
  vector<double> backward_pass(vector<double> error_from_next_layer);
  int getOutputSize();
  int getInputSize();
};
#endif // LAYER_H

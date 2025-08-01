#ifndef NEURON_H
#define NEURON_H

#include <vector>
using namespace std;

class NeuronNode
{
private:
  vector<double> weights;
  int input_size;
  double step_size = 0.001;
  double bias = 0.1;
  vector<double> last_input;
  double last_output;

public:
  NeuronNode(int input_size); // constructor
  double sigmoid(double x);
  vector<double> backward_pass(double error_from_next_layer);
  double forward_pass(vector<double> input);
};

#endif // NEURON_H

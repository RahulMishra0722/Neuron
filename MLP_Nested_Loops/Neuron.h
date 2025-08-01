#ifndef NEURON_H
#define NEURON_H

#include <vector>
using namespace std;

class Neuron
{
private:
  vector<double> weights;
  int input_size;
  double step_size = 0.01;
  double bias = 0.01; // Changed from 0.1 to 0.01
  vector<double> last_input;
  double last_output;
  double last_weighted_sum; // Store for proper derivative calculation

public:
  Neuron(int input_size);
  double sigmoid(double x);
  double sigmoid_derivative(double x); // Add proper derivative function
  vector<double> backward_pass(double error_from_next_layer);
  double forward_pass(vector<double> input);
  void setLearningRate(double lr); // Allow learning rate adjustment
};

#endif // NEURON_H

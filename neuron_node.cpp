#include "neuron_node.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <random>
using namespace std;

// Constructor implementation
NeuronNode::NeuronNode(int input_size)
{
  this->input_size = input_size;
  double xavier_range = sqrt(6.0 / 5.0);
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(-xavier_range, xavier_range);
  for (int weight = 0; weight < input_size; weight++)
  {
    weights.push_back(dis(gen));
  }
}

double NeuronNode::forward_pass(vector<double> input)
{
  double weightedSum = bias;
  last_input = input;

  for (int i = 0; i < input.size(); i++)
  {
    weightedSum += input[i] * weights[i];
  }

  double output = sigmoid(weightedSum);
  last_output = output;

  cout << fixed << setprecision(6);
  cout << "[Forward Pass]\n";
  cout << "  Weighted Sum: " << weightedSum << "\n";
  cout << "  Output (Sigmoid): " << output << "\n";

  return output;
}

vector<double> NeuronNode::backward_pass(double error_from_next_layer)
{
  vector<double> inputs_errors(weights.size());
  double error = error_from_next_layer;
  double sigmoid_deriv = last_output * (1 - last_output);
  double delta = error * sigmoid_deriv;

  // Logging backward pass
  cout << "[Backward Pass]\n";
  cout << "  Target: " << error_from_next_layer << "\n";
  cout << "  Output: " << last_output << "\n";
  cout << "  Error: " << error << "\n";
  cout << "  Sigmoid Derivative: " << sigmoid_deriv << "\n";

  for (int i = 0; i < weights.size(); i++)
  {
    double gradiant = error * sigmoid_deriv * last_input[i];
    weights[i] += step_size * gradiant;
    inputs_errors[i] = step_size * gradiant;

    if (i < 5)
    {
      cout << "  Weight[" << i << "] update: " << step_size * gradiant << "\n";
    }
  }

  bias += step_size * delta;
  cout << "  Bias update: " << step_size * delta << "\n";

  return inputs_errors;
}

double NeuronNode::sigmoid(double x)
{
  return 1.0 / (1 + exp(-x));
}

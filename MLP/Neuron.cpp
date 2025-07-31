#include "Neuron.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <random>
using namespace std;

Neuron::Neuron(int input_size)
{
  this->input_size = input_size;
  double xavier_std = sqrt(2.0 / input_size);
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dis(0.0, xavier_std);
  weights.resize(input_size);
  for (int weight = 0; weight < input_size; weight++)
  {
    weights[weight] = dis(gen);
  }
  bias = 0.01;
  cout << "Neuron initialized with " << input_size << " inputs, Xavier std: " << xavier_std << endl;
}

double Neuron::forward_pass(vector<double> input)
{
  if (input.size() != input_size)
  {
    cout << "Error: Input size mismatch! Expected " << input_size << ", got " << input.size() << endl;
    return 0.0;
  }
  double weightedSum = bias;
  last_input = input;

  for (int i = 0; i < input.size(); i++)
  {
    weightedSum += input[i] * weights[i];
  }

  double output = sigmoid(weightedSum);
  last_output = output;

  static int forward_count = 0;
  forward_count++;
  if (forward_count % 1000 == 0 || forward_count < 5)
  {
    cout << fixed << setprecision(6);
    cout << "[Forward Pass " << forward_count << "]\n";
    cout << "  Weighted Sum: " << weightedSum << "\n";
    cout << "  Output (Sigmoid): " << output << "\n";

    // Check for saturation
    if (abs(weightedSum) > 5.0)
    {
      cout << "  WARNING: Neuron saturated! (|weighted_sum| > 5.0)" << endl;
    }
  }

  return output;
}

vector<double> Neuron::backward_pass(double error_from_next_layer)
{
  vector<double> inputs_errors(input_size);
  double sigmoid_deriv = sigmoid_derivative(last_weighted_sum);
  double delta = error_from_next_layer * sigmoid_deriv;
  static int backward_count = 0;
  backward_count++;
  if (backward_count % 1000 == 0 || backward_count < 5)
  {
    cout << "[Backward Pass " << backward_count << "]\n";
    cout << "  Error from next layer: " << error_from_next_layer << "\n";
    cout << "  Last output: " << last_output << "\n";
    cout << "  Sigmoid Derivative: " << sigmoid_deriv << "\n";
    cout << "  Delta (gradient): " << delta << "\n";
  }
  for (int i = 0; i < input_size; i++)
  {
    inputs_errors[i] = delta * weights[i];
    double weight_update = step_size * delta * last_input[i];
    weights[i] -= weight_update;
    if ((backward_count % 1000 == 0 || backward_count < 5) && i < 5)
    {
      cout << "Weight[" << i
           << "] update: " << weight_update << "(new value: " << weights[i] << ")\n";
    }
  }
  double bias_update = step_size * delta;
  bias -= bias_update;
  if (backward_count % 1000 == 0 || backward_count < 5)
  {
    cout << "  Bias update: " << bias_update << " (new value: " << bias << ")\n";
  }
  return inputs_errors;
}

double Neuron::sigmoid(double x)
{
  if (x > 500)
    x = 500;
  if (x < -500)
    x = -500;
  return 1.0 / (1 + exp(-x));
}
double Neuron::sigmoid_derivative(double x)
{
  double sig = sigmoid(x);
  return sig * (1 - sig);
}

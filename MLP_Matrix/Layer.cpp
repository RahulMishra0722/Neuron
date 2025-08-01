#include "Layer.h"
#include <iostream>
#include <random>
using namespace std;

Layer::Layer(int input_size, int output_size)
{
  this->input_size = input_size;
  this->output_size = output_size;
  this->weights.resize(output_size, vector<double>(input_size));
  this->bias.resize(output_size);
  double xavier_std = sqrt(2.0 / input_size);
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dis(0.0, xavier_std);

  for (int row = 0; row < output_size; row++)
  {
    for (int col = 0; col < input_size; col++)
    {
      weights[row][col] = dis(gen);
    }
  }
}

vector<double> Layer::forward_pass(vector<double> inputs)
{
  static int forward_count = 0;
  forward_count++;

  vector<double> outputs(output_size);

  if (forward_count % 1000 == 0 || forward_count <= 5)
  {
    cout << "[Layer Forward Pass " << forward_count << "]" << endl;
    cout << "Input size: " << inputs.size() << ", First few inputs: ";
    for (int i = 0; i < min(3, (int)inputs.size()); i++)
    {
      cout << inputs[i] << " ";
    }
    cout << endl;
  }

  for (int output = 0; output < output_size; output++)
  {
    double sum = 0;
    for (int input = 0; input < input_size; input++)
    {
      sum += weights[output][input] * inputs[input];
    }
    double weighted_sum = sum + bias[output];
    outputs[output] = sigmoid(weighted_sum);

    if ((forward_count % 1000 == 0 || forward_count <= 5) && output < 3)
    {
      cout << "Output[" << output << "]: weighted_sum=" << weighted_sum
           << ", sigmoid=" << outputs[output] << endl;
    }
  }

  last_input = inputs;
  last_output = outputs;
  return outputs;
}
double Layer::sigmoid(double x)
{
  if (x > 500)
    x = 500;
  if (x < -500)
    x = -500;
  return 1.0 / (1 + exp(-x));
}

vector<double> Layer::backward_pass(vector<double> error_from_next_layer)
{
  static int backward_count = 0;
  backward_count++;

  if (backward_count % 1000 == 0 || backward_count <= 5)
  {
    cout << "[Layer Backward Pass " << backward_count << "]" << endl;
    cout << "Errors from next layer (first 3): ";
    for (int i = 0; i < min(3, (int)error_from_next_layer.size()); i++)
    {
      cout << error_from_next_layer[i] << " ";
    }
    cout << endl;
  }

  vector<double> error_from_previous_layer(input_size, 0.0);

  for (int row = 0; row < output_size; row++)
  {
    double sigmoid_deriv = last_output[row] * (1 - last_output[row]);
    double delta = error_from_next_layer[row] * sigmoid_deriv;

    if ((backward_count % 1000 == 0 || backward_count <= 5) && row < 3)
    {
      cout << "Row " << row << ": sigmoid_deriv=" << sigmoid_deriv
           << ", delta=" << delta << endl;
    }

    for (int col = 0; col < input_size; col++)
    {
      error_from_previous_layer[col] += delta * weights[row][col];
      double weight_update = step_size * delta * last_input[col];
      weights[row][col] -= weight_update;

      if ((backward_count % 1000 == 0 || backward_count <= 5) && row < 2 && col < 3)
      {
        cout << "Weight[" << row << "][" << col << "] update: " << weight_update
             << ", new value: " << weights[row][col] << endl;
      }
    }

    double bias_update = step_size * delta;
    bias[row] -= bias_update;

    if ((backward_count % 1000 == 0 || backward_count <= 5) && row < 3)
    {
      cout << "Bias[" << row << "] update: " << bias_update
           << ", new value: " << bias[row] << endl;
    }
  }

  return error_from_previous_layer;
}
int Layer::getOutputSize()
{
  return output_size;
};
int Layer::getInputSize()
{
  return input_size;
};

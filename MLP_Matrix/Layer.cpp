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

  // WEIGHT-INITIALIZATION: Use Xavier method to prevent vanishing/exploding gradients
  double xavier_std = sqrt(2.0 / input_size);
  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dis(0.0, xavier_std);

  // SETUP-CONNECTIONS: Create random weights between each input and each neuron
  for (int neuron_index = 0; neuron_index < output_size; neuron_index++)
  {
    for (int input_index = 0; input_index < input_size; input_index++)
    {
      weights[neuron_index][input_index] = dis(gen);
    }
  }
}

vector<double> Layer::forward_pass(vector<double> layer_inputs)
{
  static int forward_pass_count = 0;
  forward_pass_count++;

  vector<double> neuron_outputs(output_size);

  if (forward_pass_count % 1000 == 0 || forward_pass_count <= 5)
  {
    cout << "[Layer Forward Pass " << forward_pass_count << "]" << endl;
    cout << "Input size: " << layer_inputs.size() << ", First few inputs: ";
    for (int i = 0; i < min(3, (int)layer_inputs.size()); i++)
    {
      cout << layer_inputs[i] << " ";
    }
    cout << endl;
  }

  // NEURON-COMPUTATION: Calculate output for each neuron in this layer
  for (int current_neuron = 0; current_neuron < output_size; current_neuron++)
  {
    double weighted_sum = 0;

    // INPUT-AGGREGATION: Sum all weighted inputs for this neuron
    // MATH: weighted_sum = w₁×input₁ + w₂×input₂ + ... + wₙ×inputₙ
    for (int input_index = 0; input_index < input_size; input_index++)
    {
      weighted_sum += weights[current_neuron][input_index] * layer_inputs[input_index];
    }

    // ACTIVATION: Add bias and apply sigmoid function
    // MATH: neuron_output = sigmoid(weighted_sum + bias)
    // MATH: sigmoid(x) = 1 / (1 + e^(-x))
    double neuron_input_with_bias = weighted_sum + bias[current_neuron];
    neuron_outputs[current_neuron] = sigmoid(neuron_input_with_bias);

    if ((forward_pass_count % 1000 == 0 || forward_pass_count <= 5) && current_neuron < 3)
    {
      cout << "Neuron[" << current_neuron << "]: weighted_sum=" << neuron_input_with_bias
           << ", sigmoid_output=" << neuron_outputs[current_neuron] << endl;
    }
  }

  // MEMORY-STORAGE: Store values needed for backpropagation
  last_input = layer_inputs;
  last_output = neuron_outputs;
  return neuron_outputs;
}

double Layer::sigmoid(double neuron_input)
{
  // OVERFLOW-PROTECTION: Prevent mathematical overflow/underflow
  // MATH: sigmoid(x) = 1 / (1 + e^(-x))
  if (neuron_input > 500)
    neuron_input = 500;
  if (neuron_input < -500)
    neuron_input = -500;
  return 1.0 / (1 + exp(-neuron_input));
}

vector<double> Layer::backward_pass(vector<double> errors_from_next_layer)
{
  static int backward_pass_count = 0;
  backward_pass_count++;

  if (backward_pass_count % 1000 == 0 || backward_pass_count <= 5)
  {
    cout << "[Layer Backward Pass " << backward_pass_count << "]" << endl;
    cout << "Errors from next layer (first 3): ";
    for (int i = 0; i < min(3, (int)errors_from_next_layer.size()); i++)
    {
      cout << errors_from_next_layer[i] << " ";
    }
    cout << endl;
  }

  vector<double> errors_to_previous_layer(input_size, 0.0);

  // ERROR-PROPAGATION: Process each neuron in this layer
  for (int current_neuron = 0; current_neuron < output_size; current_neuron++)
  {
    // SENSITIVITY-CALCULATION: How sensitive this neuron's output is to input changes
    // WHY: We need to know "if I nudge the weighted sum slightly, how much does the output change?"
    // WHAT: This tells us the slope of the sigmoid curve at the current point
    // MATH: d(sigmoid(x))/dx = sigmoid(x) × (1 - sigmoid(x))
    // MATH: Since we already have sigmoid(x) stored in last_output, we can reuse it!
    double sigmoid_derivative = last_output[current_neuron] * (1 - last_output[current_neuron]);

    // GRADIENT-COMPUTATION: Calculate the error gradient for this neuron
    // WHY: We need to know "how much does the final network error change when this neuron's weighted sum changes?"
    // WHAT: This combines the error signal from ahead with our local sensitivity to get the complete picture
    // MEANING: If this value is large, small changes to this neuron's weighted sum cause big changes in final error
    // MATH: This is the chain rule in action!
    // MATH: d(final_error)/d(weighted_sum) = d(final_error)/d(neuron_output) × d(neuron_output)/d(weighted_sum)
    // MATH: errors_from_next_layer = d(final_error)/d(neuron_output) [computed by layers ahead]
    // MATH: sigmoid_derivative = d(neuron_output)/d(weighted_sum) [our local derivative]
    double neuron_error_gradient = errors_from_next_layer[current_neuron] * sigmoid_derivative;

    if ((backward_pass_count % 1000 == 0 || backward_pass_count <= 5) && current_neuron < 3)
    {
      cout << "Neuron " << current_neuron << ": sigmoid_derivative=" << sigmoid_derivative
           << ", error_gradient=" << neuron_error_gradient << endl;
    }

    // WEIGHT-AND-BLAME-UPDATE: Update weights and calculate blame for each input
    for (int input_index = 0; input_index < input_size; input_index++)
    {
      // BLAME-PASSING: Calculate how much to blame this input for the error
      // WHY: The previous layer needs to know "how much does the final error change when this input changes?"
      // WHAT: We're propagating the error signal backward so the previous layer can update their weights
      // MEANING: If this value is large, this input has a big impact on the final network error
      // MATH: This continues the chain rule one step further back
      // MATH: d(final_error)/d(input) = d(final_error)/d(weighted_sum) × d(weighted_sum)/d(input)
      // MATH: d(weighted_sum)/d(input) = weight [since weighted_sum = w₁×input₁ + w₂×input₂ + ...]
      errors_to_previous_layer[input_index] += neuron_error_gradient * weights[current_neuron][input_index];

      // SELF-CORRECTION: Update the weight based on the error and input value
      // WHY: We want to adjust this weight to reduce the final network error
      // WHAT: We're moving the weight in the direction that reduces error (opposite to gradient)
      // MEANING: If the gradient is positive, increasing this weight increases error, so we decrease it
      // MATH: This completes the chain rule for weight updates
      // MATH: d(final_error)/d(weight) = d(final_error)/d(weighted_sum) × d(weighted_sum)/d(weight)
      // MATH: d(weighted_sum)/d(weight) = input [since weighted_sum = w₁×input₁ + w₂×input₂ + ...]
      // MATH: weight_adjustment = learning_rate × d(final_error)/d(weight)
      double weight_adjustment = step_size * neuron_error_gradient * last_input[input_index];
      weights[current_neuron][input_index] -= weight_adjustment; // Move opposite to gradient (gradient descent)

      if ((backward_pass_count % 1000 == 0 || backward_pass_count <= 5) && current_neuron < 2 && input_index < 3)
      {
        cout << "Weight[neuron" << current_neuron << "][input" << input_index
             << "] adjustment: " << weight_adjustment
             << ", new_weight: " << weights[current_neuron][input_index] << endl;
      }
    }

    // BIAS-CORRECTION: Update bias for this neuron
    // WHY: Bias affects the weighted sum directly, so we need to adjust it to reduce error
    // WHAT: We're shifting the neuron's activation threshold up or down
    // MEANING: If gradient is positive, increasing bias increases error, so we decrease it
    // MATH: d(final_error)/d(bias) = d(final_error)/d(weighted_sum) × d(weighted_sum)/d(bias)
    // MATH: d(weighted_sum)/d(bias) = 1 [since weighted_sum = w₁×input₁ + ... + bias]
    // MATH: So bias adjustment = learning_rate × neuron_error_gradient × 1 = learning_rate × neuron_error_gradient
    double bias_adjustment = step_size * neuron_error_gradient;
    bias[current_neuron] -= bias_adjustment; // Move opposite to gradient (gradient descent)

    if ((backward_pass_count % 1000 == 0 || backward_pass_count <= 5) && current_neuron < 3)
    {
      cout << "Bias[neuron" << current_neuron << "] adjustment: " << bias_adjustment
           << ", new_bias: " << bias[current_neuron] << endl;
    }
  }

  return errors_to_previous_layer;
}

int Layer::getOutputSize()
{
  return output_size;
}

int Layer::getInputSize()
{
  return input_size;
}

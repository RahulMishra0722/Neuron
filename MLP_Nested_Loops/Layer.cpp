// #include "Layer.h"
// #include "Neuron.h"
// #include <iostream>
// using namespace std;

// Layer::Layer(int input_size, int output_size)
// {
//   this->input_size = input_size;
//   this->output_size = output_size;
//   for (int i = 0; i < output_size; i++)
//   {
//     neurons.push_back(Neuron(input_size));
//   }
// }
// vector<double> Layer::forward_pass(vector<double> input)
// {

//   vector<double> outputs;

//   for (int i = 0; i < neurons.size(); i++)
//   {
//     double neuron_output = neurons[i].forward_pass(input);
//     outputs.push_back(neuron_output);
//   }
//   return outputs;
// };
// vector<double> Layer::backward_pass(vector<double> error_from_next_layer)
// {
//   vector<double> error_from_previous_layer(input_size, 0.0);
//   for (int i = 0; i < neurons.size(); i++)
//   {
//     vector<double> neuron_errors = neurons[i].backward_pass(error_from_next_layer[i]);
//     for (int j = 0; j < neuron_errors.size(); j++)
//     {
//       error_from_previous_layer[j] += neuron_errors[j];
//     }
//   }
//   return error_from_previous_layer;
// };
// int Layer::getOutputSize()
// {
//   return output_size;
// };
// int Layer::getInputSize()
// {
//   return input_size;
// };

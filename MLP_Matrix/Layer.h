#ifndef LAYER_H
#define LAYER_H
#include <iostream>
using namespace std;
class Layer
{
private:
  vector<vector<double>> weights;
  vector<double> bias;
  int input_size;
  int output_size;
  double step_size = 0.01;
  vector<double> last_input;
  vector<double> last_output;

public:
  Layer(int input_size, int output_size);
  vector<double> forward_pass(vector<double> input);
  vector<double> backward_pass(vector<double> error_from_next_layer);
  double sigmoid(double x);
  double sigmoid_derivative(double x); // Add proper derivative function
  int getOutputSize();
  int getInputSize();
};
#endif // LAYER_H

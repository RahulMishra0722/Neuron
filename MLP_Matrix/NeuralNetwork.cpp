#include "NeuralNetwork.h"
#include "Layer.h"
#include <algorithm>
#include <cmath>

NeuralNetwork::NeuralNetwork()
{
}

void NeuralNetwork::addLayer(int input_size, int output_size)
{
  layers.push_back(Layer(input_size, output_size));
}

void NeuralNetwork::train(vector<vector<double>> &training_data, vector<int> &labels, int epochs)
{
  // Split data into training and validation sets (80/20 split)
  int validation_size = training_data.size() * 0.2;
  int training_size = training_data.size() - validation_size;

  cout << "Training samples: " << training_size << ", Validation samples: " << validation_size << endl;

  for (int epoch = 0; epoch < epochs; epoch++)
  {
    double total_loss = 0.0;
    int correct_predictions = 0;

    // Training phase
    for (int i = 0; i < training_size; i++)
    {
      vector<double> prediction = forward_pass(training_data[i]);
      vector<double> target = createOneHotVector(labels[i], 10);

      // Calculate loss
      double sample_loss = calculateLoss(prediction, target);
      total_loss += sample_loss;

      // Check if prediction is correct
      if (getPredictedClass(prediction) == labels[i])
      {
        correct_predictions++;
      }

      // Backpropagation
      backward_pass(target);
    }

    // Calculate training accuracy and average loss
    double training_accuracy = (double)correct_predictions / training_size * 100.0;
    double avg_loss = total_loss / training_size;

    // Validation phase (every 10 epochs or last epoch)
    if (epoch % 10 == 0 || epoch == epochs - 1)
    {
      double validation_accuracy = validateModel(training_data, labels, training_size, validation_size);

      cout << "Epoch " << epoch + 1 << "/" << epochs
           << " - Loss: " << avg_loss
           << " - Training Accuracy: " << training_accuracy << "%"
           << " - Validation Accuracy: " << validation_accuracy << "%" << endl;
    }
    else
    {
      cout << "Epoch " << epoch + 1 << "/" << epochs
           << " - Loss: " << avg_loss
           << " - Training Accuracy: " << training_accuracy << "%" << endl;
    }
  }
}

double NeuralNetwork::validateModel(vector<vector<double>> &data, vector<int> &labels,
                                    int training_size, int validation_size)
{
  int correct_predictions = 0;

  for (int i = training_size; i < training_size + validation_size; i++)
  {
    vector<double> prediction = forward_pass(data[i]);
    int predicted_class = getPredictedClass(prediction);

    if (predicted_class == labels[i])
    {
      correct_predictions++;
    }
  }

  return (double)correct_predictions / validation_size * 100.0;
}

double NeuralNetwork::testModel(vector<vector<double>> &test_data, vector<int> &test_labels)
{
  int correct_predictions = 0;
  vector<vector<int>> confusion_matrix(10, vector<int>(10, 0));

  cout << "\nTesting model on " << test_data.size() << " samples..." << endl;

  for (int i = 0; i < test_data.size(); i++)
  {
    vector<double> prediction = forward_pass(test_data[i]);
    int predicted_class = getPredictedClass(prediction);
    int actual_class = test_labels[i];

    confusion_matrix[actual_class][predicted_class]++;

    if (predicted_class == actual_class)
    {
      correct_predictions++;
    }

    // Print some sample predictions for debugging
    if (i < 10)
    {
      cout << "Sample " << i << " - Actual: " << actual_class
           << ", Predicted: " << predicted_class
           << ", Confidence: " << prediction[predicted_class] << endl;
    }
  }

  double test_accuracy = (double)correct_predictions / test_data.size() * 100.0;

  cout << "\n=== FINAL RESULTS ===" << endl;
  cout << "Test Accuracy: " << test_accuracy << "%" << endl;
  cout << "Correct predictions: " << correct_predictions << "/" << test_data.size() << endl;

  // Print confusion matrix
  printConfusionMatrix(confusion_matrix);

  return test_accuracy;
}

void NeuralNetwork::printConfusionMatrix(vector<vector<int>> &matrix)
{
  cout << "\nConfusion Matrix:" << endl;
  cout << "Actual\\Predicted ";
  for (int i = 0; i < 10; i++)
    cout << i << "  ";
  cout << endl;

  for (int i = 0; i < 10; i++)
  {
    cout << "     " << i << "        ";
    for (int j = 0; j < 10; j++)
    {
      cout << matrix[i][j] << "  ";
    }
    cout << endl;
  }
}

vector<double> NeuralNetwork::createOneHotVector(int label, int num_classes)
{
  vector<double> oneHot(num_classes, 0.0);
  oneHot[label] = 1.0;
  return oneHot;
}

int NeuralNetwork::getPredictedClass(vector<double> &output)
{
  return max_element(output.begin(), output.end()) - output.begin();
}

vector<double> NeuralNetwork::backward_pass(vector<double> target)
{
  vector<double> errors(target.size());
  for (int output = 0; output < target.size(); output++)
  {
    errors[output] = last_network_output[output] - target[output];
  }
  for (int layer = layers.size() - 1; layer >= 0; layer--)
  {
    // what this does is the next layer will fix its own erros / what it contributed to the error and will reduce its weights and do gradiant decent and now it sorta points out the errors
    errors = layers[layer].backward_pass(errors);
  }
  return errors;
}

vector<double> NeuralNetwork::forward_pass(vector<double> input)
{
  vector<double> current_input = input;
  for (int layer = 0; layer < layers.size(); layer++)
  {
    current_input = layers[layer].forward_pass(current_input);
  }
  last_network_output = current_input;
  return current_input;
}

double NeuralNetwork::calculateLoss(vector<double> predicted, vector<double> targets)
{
  // Cross-entropy loss for classification
  double loss = 0.0;
  for (int i = 0; i < predicted.size(); i++)
  {
    // Add small epsilon to prevent log(0)
    double epsilon = 1e-15;
    double pred = max(epsilon, min(1.0 - epsilon, predicted[i]));
    loss -= targets[i] * log(pred);
  }
  return loss;
}

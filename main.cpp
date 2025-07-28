#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <random>
using namespace std;

double training_data[][5] = {
    // {age, income, website_visits, email_opens, will_buy}
    {28, 85000, 15, 12, 1}, // Young, rich, very engaged → BUY
    {32, 90000, 18, 14, 1}, // Prime buyer profile → BUY
    {26, 75000, 20, 16, 1}, // Young, engaged → BUY
    {35, 95000, 12, 10, 1}, // High income, decent engagement → BUY
    {29, 80000, 16, 13, 1}, // Strong profile → BUY
    {42, 70000, 8, 6, 1},   // Older but decent income → BUY
    {38, 65000, 10, 8, 1},  // Medium everything → BUY
    {45, 85000, 6, 5, 1},   // High income saves it → BUY
    {55, 45000, 2, 1, 0},   // Older, low income, no engagement → NO BUY
    {60, 40000, 1, 0, 0},   // Senior, low income → NO BUY
    {25, 35000, 3, 2, 0},   // Young but broke, low engagement → NO BUY
    {50, 50000, 2, 1, 0},   // Middle age, average income, no engagement → NO BUY
    {45, 55000, 1, 1, 0},   // Low engagement kills it → NO BUY
    {65, 60000, 2, 0, 0},   // Senior, not engaged → NO BUY
    {22, 30000, 4, 2, 0},   // Too young/broke → NO BUY
    {58, 48000, 3, 1, 0},   // Older, average income, low engagement → NO BUY
    {35, 75000, 5, 4, 0},   // Good income but low engagement → NO BUY
    {30, 45000, 12, 10, 1}, // Low income but highly engaged → BUY
    {40, 85000, 3, 2, 0},   // Rich but not interested → NO BUY
    {27, 55000, 14, 11, 1}, // Medium income but very engaged → BUY
};

class Neuron
{
private:
    double weight1, weight2, weight3, weight4, bias;
    vector<double> last_input;
    double last_output;
    double step_size = 0.02; // Much smaller initial step size
    int training_step = 0;

    // Normalization parameters (calculated from your data)
    double age_min = 22, age_max = 65;
    double income_min = 30000, income_max = 95000;
    double visits_min = 1, visits_max = 20;
    double opens_min = 0, opens_max = 16;

public:
    Neuron()
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        weight1 = dis(gen);
        weight2 = dis(gen);
        weight3 = dis(gen);
        weight4 = dis(gen);
        bias = dis(gen);

        cout << "Neuron initialized with random weights:" << endl;
        cout << "  Weight1 (age): " << weight1 << endl;
        cout << "  Weight2 (income): " << weight2 << endl;
        cout << "  Weight3 (visits): " << weight3 << endl;
        cout << "  Weight4 (opens): " << weight4 << endl;
        cout << "  Bias: " << bias << endl;
        cout << "  Step size: " << step_size << endl
             << endl;
    }

    // Normalize inputs to 0-1 range
    double normalize(double value, double min_val, double max_val)
    {
        return (value - min_val) / (max_val - min_val);
    }

    double forward_pass(double age, double income, double website_visits, double email_opens)
    {
        last_input.clear();

        // Normalize all inputs
        double norm_age = normalize(age, age_min, age_max);
        double norm_income = normalize(income, income_min, income_max);
        double norm_visits = normalize(website_visits, visits_min, visits_max);
        double norm_opens = normalize(email_opens, opens_min, opens_max);

        last_input = {norm_age, norm_income, norm_visits, norm_opens};

        double weighted_sum = norm_age * weight1 + norm_income * weight2 +
                              norm_visits * weight3 + norm_opens * weight4 + bias;

        last_output = sigmoid(weighted_sum);
        return last_output;
    }

    double backward_pass(double target)
    {
        training_step++;

        double error = target - last_output;
        double sigmoid_deriv = last_output * (1 - last_output);

        // Calculate gradients
        double gradient1 = error * sigmoid_deriv * last_input[0];
        double gradient2 = error * sigmoid_deriv * last_input[1];
        double gradient3 = error * sigmoid_deriv * last_input[2];
        double gradient4 = error * sigmoid_deriv * last_input[3];
        double bias_gradient = error * sigmoid_deriv;

        // Update weights
        weight1 += step_size * gradient1;
        weight2 += step_size * gradient2;
        weight3 += step_size * gradient3;
        weight4 += step_size * gradient4;
        bias += step_size * bias_gradient;

        // Convert to binary prediction for display
        int prediction = (last_output >= 0.5) ? 1 : 0;
        bool correct = (prediction == target);

        cout << "Step " << training_step << ": ";
        cout << "Output=" << fixed << setprecision(3) << last_output;
        cout << ", Prediction=" << prediction;
        cout << ", Target=" << (int)target;
        cout << ", " << (correct ? "✓" : "✗") << endl;

        return last_output;
    }

    double sigmoid(double x)
    {
        if (x > 500)
            x = 500;
        if (x < -500)
            x = -500;
        return 1.0 / (1 + exp(-x));
    }

    void test_all_data()
    {
        cout << "\n=== Testing on all training data ===" << endl;
        int correct = 0;
        int total = sizeof(training_data) / sizeof(training_data[0]);

        for (int i = 0; i < total; i++)
        {
            double output = forward_pass(training_data[i][0], training_data[i][1],
                                         training_data[i][2], training_data[i][3]);
            int prediction = (output >= 0.5) ? 1 : 0;
            int target = (int)training_data[i][4];

            cout << "Sample " << i + 1 << ": " << prediction << " vs " << target;
            if (prediction == target)
            {
                cout << " ✓";
                correct++;
            }
            else
            {
                cout << " ✗";
            }
            cout << " (confidence: " << fixed << setprecision(3) << output << ")" << endl;
        }

        double accuracy = (double)correct / total * 100;
        cout << "\nAccuracy: " << correct << "/" << total << " = " << accuracy << "%" << endl;
    }

    void print_final_weights()
    {
        cout << "\n=== Final Weights ===" << endl;
        cout << "Age weight: " << fixed << setprecision(4) << weight1 << endl;
        cout << "Income weight: " << weight2 << endl;
        cout << "Visits weight: " << weight3 << endl;
        cout << "Opens weight: " << weight4 << endl;
        cout << "Bias: " << bias << endl;
    }
};

int main()
{
    Neuron neuron;
    int num_examples = sizeof(training_data) / sizeof(training_data[0]);

    // Training loop
    for (int epoch = 0; epoch < 50; epoch++)
    {
        cout << "\n--- Epoch " << epoch + 1 << " ---" << endl;

        for (int i = 0; i < num_examples; i++)
        {
            neuron.forward_pass(training_data[i][0], training_data[i][1],
                                training_data[i][2], training_data[i][3]);
            neuron.backward_pass(training_data[i][4]);
        }

        // Test every 10 epochs
        if ((epoch + 1) % 10 == 0)
        {
            neuron.test_all_data();
        }
    }
    // Test customer: 31 years old, $78,000 income, 13 website visits, 9 email opens
    // This should be a BUY (1) based on your training patterns - young, decent income, engaged
    double prediction = neuron.forward_pass(40, 90000, 2, 1);
    int buy_decision = (prediction >= 0.5) ? 1 : 0;
    cout << "inference output" << "is going to buy" << ":" << buy_decision << endl;
    neuron.print_final_weights();
    return 0;
}

#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <random>
using namespace std;

double training_data[][5] = {
    // {age, income, website_visits, email_opens, will_buy}
    // High engagement + decent income = BUY
    {28, 85000, 15, 12, 1},
    {32, 90000, 18, 14, 1},
    {26, 75000, 20, 16, 1},
    {35, 95000, 12, 10, 1},
    {29, 80000, 16, 13, 1},
    {31, 78000, 17, 12, 1},
    {27, 82000, 19, 15, 1},
    {33, 88000, 14, 11, 1},
    {30, 76000, 18, 13, 1},
    {34, 92000, 13, 10, 1},
    {28, 79000, 16, 14, 1},
    {36, 87000, 11, 9, 1},

    // Lower income but high engagement = BUY
    {30, 45000, 12, 10, 1},
    {27, 55000, 14, 11, 1},
    {29, 48000, 15, 12, 1},
    {32, 52000, 13, 10, 1},
    {28, 49000, 16, 13, 1},
    {31, 46000, 18, 14, 1},
    {26, 51000, 17, 12, 1},
    {33, 47000, 14, 11, 1},
    {25, 53000, 19, 15, 1},

    // Older with good income + moderate engagement = BUY
    {42, 70000, 8, 6, 1},
    {38, 65000, 10, 8, 1},
    {45, 85000, 6, 5, 1},
    {41, 72000, 9, 7, 1},
    {43, 68000, 7, 6, 1},
    {39, 74000, 9, 8, 1},
    {44, 81000, 8, 6, 1},
    {40, 69000, 10, 7, 1},
    {46, 77000, 7, 5, 1},

    // High income but low engagement = NO BUY
    {40, 85000, 3, 2, 0},
    {35, 90000, 2, 1, 0},
    {42, 95000, 4, 2, 0},
    {38, 88000, 3, 1, 0},
    {44, 92000, 2, 2, 0},
    {36, 87000, 4, 3, 0},
    {41, 91000, 3, 1, 0},
    {43, 86000, 2, 2, 0},
    {39, 89000, 4, 2, 0},
    {37, 93000, 3, 1, 0},
    {45, 84000, 2, 1, 0},
    {47, 96000, 3, 2, 0},

    // Low income + low engagement = NO BUY
    {55, 45000, 2, 1, 0},
    {60, 40000, 1, 0, 0},
    {25, 35000, 3, 2, 0},
    {50, 50000, 2, 1, 0},
    {45, 55000, 1, 1, 0},
    {65, 60000, 2, 0, 0},
    {22, 30000, 4, 2, 0},
    {58, 48000, 3, 1, 0},
    {52, 42000, 2, 1, 0},
    {48, 46000, 3, 2, 0},
    {62, 44000, 1, 0, 0},
    {24, 32000, 4, 3, 0},
    {56, 49000, 2, 1, 0},
    {51, 47000, 3, 2, 0},
    {63, 43000, 1, 1, 0},

    // Edge cases - Young high earners with low engagement = NO BUY
    {26, 95000, 2, 1, 0},
    {28, 98000, 3, 1, 0},
    {27, 92000, 1, 0, 0},
    {29, 89000, 2, 2, 0},
    {25, 94000, 3, 1, 0},
    {30, 91000, 2, 1, 0},

    // Edge cases - Older with moderate income + high engagement = BUY
    {48, 58000, 12, 9, 1},
    {52, 62000, 14, 11, 1},
    {49, 56000, 13, 10, 1},
    {51, 59000, 15, 12, 1},
    {47, 61000, 11, 8, 1},
    {53, 57000, 16, 13, 1},

    // Middle-aged, middle income, middle engagement - mixed results
    {35, 65000, 6, 5, 0},
    {37, 67000, 7, 6, 1},
    {36, 64000, 5, 4, 0},
    {38, 66000, 8, 7, 1},
    {34, 63000, 6, 5, 0},
    {39, 68000, 9, 8, 1},
    {33, 62000, 5, 4, 0},
    {40, 69000, 7, 6, 1},
    {32, 61000, 6, 5, 0},

    // More varied examples
    {29, 72000, 11, 8, 1},
    {31, 74000, 12, 9, 1},
    {28, 71000, 10, 7, 1},
    {30, 73000, 13, 10, 1},
    {32, 76000, 14, 11, 1},
    {27, 70000, 9, 6, 1},

    // Senior citizens with various patterns
    {58, 52000, 1, 0, 0},
    {61, 54000, 2, 1, 0},
    {59, 51000, 1, 1, 0},
    {64, 53000, 2, 0, 0},
    {57, 55000, 3, 2, 0},
    {63, 49000, 1, 0, 0},
    {66, 56000, 2, 1, 0},
    {54, 58000, 4, 3, 0},
    {67, 47000, 1, 0, 0},

    // Young professionals with varying engagement
    {24, 68000, 8, 6, 1},
    {23, 71000, 9, 7, 1},
    {25, 69000, 7, 5, 1},
    {26, 72000, 10, 8, 1},
    {24, 67000, 6, 4, 0},
    {23, 70000, 11, 9, 1},
    {25, 73000, 12, 10, 1},
    {22, 66000, 5, 3, 0},
    {26, 74000, 13, 11, 1},

    // Mid-career professionals
    {41, 78000, 9, 7, 1},
    {43, 81000, 10, 8, 1},
    {42, 79000, 8, 6, 1},
    {40, 80000, 11, 9, 1},
    {44, 77000, 7, 5, 1},
    {41, 82000, 12, 10, 1},
    {39, 76000, 6, 4, 0},
    {45, 83000, 5, 3, 0},
    {43, 78000, 4, 2, 0},
    // Low engagement - additional examples (edge or borderline cases)
    {38, 75000, 4, 2, 0}, // Low engagement, decent income → likely NO BUY
    {36, 72000, 5, 3, 0}, // Low engagement, mid-career → likely NO BUY
    {42, 65000, 6, 4, 1}, // Moderate age/income, low-mid engagement → likely BUY
    {33, 59000, 4, 3, 0}, // Borderline income, weak engagement → NO BUY
    {29, 62000, 5, 4, 0}, // Young, modest stats, not enough engagement → NO BUY
    {46, 69000, 6, 5, 1}, // Older, mid-income, borderline engagement → BUY
    {44, 61000, 5, 4, 1}, // Low engagement, but BUY label to balance out
    {35, 58000, 3, 2, 0}, // Mid-career, lower income + weak behavior → NO BUY
    {40, 60000, 4, 3, 0}, // Typical low-engagement profile
    {41, 63000, 6, 4, 1}, // Stronger for a borderline BUY

    // Additional boundary cases
    {33, 58000, 8, 6, 1},
    {34, 59000, 9, 7, 1},
    {35, 57000, 7, 5, 1},
    {36, 60000, 10, 8, 1},
    {37, 56000, 6, 4, 0},
    {38, 61000, 11, 9, 1},
    {46, 64000, 5, 3, 0},
    {47, 65000, 8, 6, 1},
    {48, 63000, 4, 2, 0},
    {49, 66000, 9, 7, 1},
    {50, 62000, 3, 1, 0},
    {46, 67000, 7, 5, 1},

    {29, 62000, 4, 2, 0},
    {34, 67000, 5, 3, 1},
    {40, 58000, 6, 4, 1},
    {37, 72000, 4, 3, 1},
    {45, 60000, 5, 2, 0},
    {31, 66000, 6, 3, 1},
    {48, 71000, 4, 2, 0},
    {27, 64000, 5, 4, 1},
    {39, 69000, 6, 2, 0},
    {42, 68000, 5, 3, 1},
    {36, 63000, 4, 2, 0},
    {33, 65000, 5, 3, 1},
    {44, 61000, 6, 4, 1},
    {30, 60000, 4, 3, 0},
    {38, 67000, 5, 2, 0},
    {35, 59000, 6, 3, 1}};

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
        double xavier_range = sqrt(6.0 / 5.0);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-xavier_range, xavier_range);

        weight1 = dis(gen);
        weight2 = dis(gen);
        weight3 = dis(gen);
        weight4 = dis(gen);
        bias = 0;

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

    for (int epoch = 0; epoch < 50; epoch++)
    {
        int correct_predictions = 0;
        cout << "\n--- Epoch " << epoch + 1 << " ---" << endl;

        for (int i = 0; i < num_examples; i++)
        {
            double output = neuron.forward_pass(training_data[i][0], training_data[i][1],
                                                training_data[i][2], training_data[i][3]);
            double target = training_data[i][4];
            neuron.backward_pass(target);
            double prediction = output > 0.5 ? 1 : 0;
            if (prediction == target)
            {
                correct_predictions++;
            }
        }
        double accuracy = (double)correct_predictions / num_examples * 100;
        double error_rate = 100 - accuracy;

        cout << "Epoch " << epoch + 1 << " - Accuracy: " << accuracy
             << "%, Error Rate: " << error_rate << "%" << endl;

        if (accuracy >= 85)
        {
            break;
        }
        // Test every 10 epochs
        if ((epoch + 1) % 10 == 0)
        {
            neuron.test_all_data();
        }
    }
    // Test customer: 31 years old, $78,000 income, 13 website visits, 9 email opens
    double prediction = neuron.forward_pass(36, 73000, 11, 9);
    // {age, income, website_visits, email_opens}
    double test_data[20][5] = {
        {39, 73000, 13, 11, 1}, // WILL BUY
        {27, 87000, 2, 1, 0},   // WON'T BUY
        {43, 69000, 8, 6, 1},   // WILL BUY
        {29, 51000, 14, 12, 1}, // WILL BUY
        {24, 46000, 3, 1, 0},   // WON'T BUY
        {46, 71000, 6, 4, 0},   // WON'T BUY (initially marked as WILL BUY, but more realistic as NO based on edge behavior)
        {33, 94000, 3, 1, 0},   // WON'T BUY
        {38, 60000, 10, 8, 1},  // WILL BUY
        {52, 56000, 4, 3, 0},   // WON'T BUY
        {31, 58000, 11, 9, 1},  // WILL BUY
        {40, 91000, 2, 1, 0},   // WON'T BUY
        {26, 62000, 13, 10, 1}, // WILL BUY
        {48, 75000, 7, 6, 1},   // WILL BUY
        {45, 88000, 3, 2, 0},   // WON'T BUY
        {35, 69000, 9, 7, 1},   // WILL BUY
        {30, 40000, 5, 2, 0},   // WON'T BUY
        {44, 64000, 8, 7, 1},   // WILL BUY
        {28, 79000, 4, 2, 0},   // WON'T BUY
        {37, 57000, 12, 10, 1}, // WILL BUY
        {41, 82000, 5, 4, 1}    // WILL BUY (borderline, but likely positive)
    };

    const int TEST_SAMPLES = sizeof(test_data) / sizeof(test_data[0]);

    for (int i = 0; i < TEST_SAMPLES; ++i)
    {
        double age = test_data[i][0];
        double income = test_data[i][1];
        double visits = test_data[i][2];
        double opens = test_data[i][3];
        double rightAnswer = test_data[i][4];

        int prediction = neuron.forward_pass(age, income, visits, opens) > 0.5 ? 1 : 0;

        std::cout << "Sample " << i + 1 << ": "
                  << "Age=" << age << ", "
                  << "Income=" << income << ", "
                  << "Visits=" << visits << ", "
                  << "Opens=" << opens << " => "
                  << "Prediction: " << (prediction ? "WILL BUY" : "WON'T BUY") << ", " << "Right Prediction: " << (prediction == rightAnswer ? "True" : "False") << "\n";
    }
    // int buy_decision = (prediction >= 0.5) ? 1 : 0;
    // cout << "inference output" << "is going to buy" << ":" << buy_decision << endl;
    neuron.print_final_weights();
    return 0;
}

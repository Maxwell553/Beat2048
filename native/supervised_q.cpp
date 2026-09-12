// Train the direct current-board n-tuple action network from frozen RL targets.
#define RL_LIBRARY
#include "rl.cpp"
#include <algorithm>
#include <cmath>
#include <iomanip>

struct LabelRecord {
    uint64_t board;
    float q[4];
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: supervised_q DATA OUTPUT EPOCHS [ALPHA] [RESUME|-] [classification|regression]\n";
        return 2;
    }
    const std::string data_file = argv[1];
    const std::string output_file = argv[2];
    const int epochs = std::stoi(argv[3]);
    const float initial_alpha = argc > 4 ? std::stof(argv[4]) : 0.02f;
    const bool regression = argc > 6 && std::string(argv[6]) == "regression";
    Network network(0);
    if (argc > 5 && std::string(argv[5]) != "-") network.load(argv[5]);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::ifstream input(data_file, std::ios::binary);
        if (!input) throw std::runtime_error("could not open training data");
        const float alpha = initial_alpha *
            std::pow(0.5f, float(epoch - 1) / std::max(1, epochs - 1));
        uint64_t records = 0, correct = 0;
        double loss = 0;
        LabelRecord record;
        while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            float logits[4], largest = -1e30f;
            int target = -1;
            for (int action = 0; action < 4; ++action) {
                logits[action] = network.value(record.board, action);
                if (record.q[action] > largest) {
                    largest = record.q[action];
                    target = action;
                }
            }

            float maximum = -1e30f;
            int predicted = -1, legal_count = 0;
            float target_mean = 0;
            for (int action = 0; action < 4; ++action) {
                if (record.q[action] <= -1e20f) continue;
                target_mean += record.q[action];
                ++legal_count;
                if (predicted < 0 || logits[action] > maximum) {
                    maximum = logits[action];
                    predicted = action;
                }
            }
            correct += predicted == target;

            if (regression) {
                target_mean /= legal_count;
                for (int action = 0; action < 4; ++action) {
                    if (record.q[action] <= -1e20f) continue;
                    const float wanted = (record.q[action] - target_mean) / 1000.0f;
                    const float error = wanted - logits[action];
                    loss += error * error;
                    network.update(record.board, action, error, alpha);
                }
            } else {
                float denominator = 0;
                for (int action = 0; action < 4; ++action) {
                    if (record.q[action] > -1e20f)
                        denominator += std::exp(std::min(40.0f, logits[action] - maximum));
                }
                loss -= std::log(std::max(1e-12f,
                    std::exp(std::min(40.0f, logits[target] - maximum)) / denominator));
                for (int action = 0; action < 4; ++action) {
                    if (record.q[action] <= -1e20f) continue;
                    const float probability =
                        std::exp(std::min(40.0f, logits[action] - maximum)) / denominator;
                    network.update(record.board, action,
                        (action == target ? 1.0f : 0.0f) - probability, alpha);
                }
            }
            ++records;
        }
        network.save(output_file + ".epoch" + std::to_string(epoch));
        network.save(output_file);
        std::cout << std::setprecision(8)
                  << "{\"epoch\":" << epoch
                  << ",\"records\":" << records
                  << ",\"alpha\":" << alpha
                  << ",\"mode\":\"" << (regression ? "regression" : "classification") << "\""
                  << ",\"online_loss\":" << loss / records
                  << ",\"online_accuracy\":" << double(correct) / records
                  << "}" << std::endl;
    }
}

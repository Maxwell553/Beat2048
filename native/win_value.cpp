// Training-only afterstate critic for probability of reaching 2048.
#define RL_LIBRARY
#include "rl.cpp"

struct ValueNetwork : Network {
    using Network::Network;
    float evaluate(Board board) const {
        float total = 0;
        for (int symmetry = 0; symmetry < 4; ++symmetry) total += value(board, symmetry);
        return total * .25f;
    }
    void learn(Board board, float error, float alpha) {
        size_t ids[32];
        for (int symmetry = 0; symmetry < 4; ++symmetry) indices(board, symmetry, ids + symmetry * ACTIVE);
        const float step = alpha * error / ACTIVE / 4;
        for (auto id : ids) weights[id] += step;
    }
};

struct Step { Board after; float value; };

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "train";
        const int games = argc > 2 ? std::stoi(argv[2]) : 100000;
        const std::string output = argc > 3 ? argv[3] : "models/rl/win_teacher.bin";
        const uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 100000000;
        const std::string behavior_file = argc > 5 ? argv[5] : "models/rl/value_teacher.bin";
        const float alpha = argc > 6 ? std::stof(argv[6]) : .05f;
        const float lambda = argc > 7 ? std::stof(argv[7]) : .95f;
        const float epsilon = argc > 8 ? std::stof(argv[8]) : .03f;
        const float blend = argc > 9 ? std::stof(argv[9]) : .2f;
        ValueNetwork network(.5f), behavior(0); Moves moves;
        behavior.load(behavior_file);
        if (command == "resume" || command == "eval") network.load(output);
        if (command == "train") network.save(output + ".initial");
        auto started = std::chrono::steady_clock::now();
        int wins = 0, losses = 0; int64_t steps = 0;
        for (int episode = 1; episode <= games; ++episode) {
            std::mt19937_64 rng(seed + episode - 1); Board board = spawn(spawn(0, rng), rng);
            std::vector<Step> trajectory; trajectory.reserve(2000); bool won = false;
            for (int count = 0; count < 30000; ++count) {
                int mask = moves.mask(board); if (!mask) break;
                int action = -1; float best = -1e30f; float score_values[4] = {}, win_values[4] = {};
                std::vector<int> legal;
                for (int candidate = 0; candidate < 4; ++candidate) if (mask & (1 << candidate)) {
                    legal.push_back(candidate); int reward; Board after = moves.move(board, candidate, reward);
                    score_values[candidate] = reward + behavior.evaluate(after);
                    win_values[candidate] = network.evaluate(after);
                    float q = score_values[candidate];
                    if (rank_max(after) >= 11) q = 1e30f;
                    if (q > best) { best = q; action = candidate; }
                }
                if (command == "eval") {
                    float sm = 0, wm = 0;
                    for (int candidate : legal) { sm += score_values[candidate]; wm += win_values[candidate]; }
                    sm /= legal.size(); wm /= legal.size(); float ss = 1e-6f, ws = 1e-6f;
                    for (int candidate : legal) { ss += (score_values[candidate]-sm)*(score_values[candidate]-sm); ws += (win_values[candidate]-wm)*(win_values[candidate]-wm); }
                    ss = std::sqrt(ss/legal.size()); ws = std::sqrt(ws/legal.size()); best = -1e30f;
                    for (int candidate : legal) {
                        int reward; Board after = moves.move(board,candidate,reward);
                        float q = (1-blend)*(score_values[candidate]-sm)/ss + blend*(win_values[candidate]-wm)/ws;
                        if (rank_max(after) >= 11) q = 1e30f;
                        if (q > best) { best = q; action = candidate; }
                    }
                }
                if (command != "eval" && std::uniform_real_distribution<float>(0, 1)(rng) < epsilon)
                    action = legal[std::uniform_int_distribution<int>(0, int(legal.size()) - 1)(rng)];
                int reward; Board after = moves.move(board, action, reward);
                trajectory.push_back({after, network.evaluate(after)}); ++steps;
                if (rank_max(after) >= 11) { won = true; break; }
                board = spawn(after, rng);
            }
            if (command != "eval") {
                float target = won ? 1.f : 0.f;
                for (auto it = trajectory.rbegin(); it != trajectory.rend(); ++it) {
                    network.learn(it->after, target - network.evaluate(it->after), alpha);
                    target = lambda * target + (1 - lambda) * it->value;
                }
            }
            wins += won; losses += !won;
            if (command == "eval") std::cout << "{\"seed\":" << seed + episode - 1 << ",\"won\":" << (won ? "true" : "false") << ",\"moves\":" << trajectory.size() << "}\n";
            if (episode % 1000 == 0 || episode == games) {
                const int divisor = episode % 1000 ? episode % 1000 : 1000;
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                std::cerr << "{\"episodes\":" << episode << ",\"window_games\":" << divisor << ",\"wins\":" << wins << ",\"losses\":" << losses << ",\"mean_moves\":" << double(steps) / divisor << ",\"seconds\":" << seconds << "}\n";
                wins = losses = 0; steps = 0;
            }
            if (command != "eval" && (episode % 10000 == 0 || episode == games)) network.save(output);
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

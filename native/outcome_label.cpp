// Label current boards by complete-game outcomes under a frozen learned RL continuation.
#define RL_LIBRARY
#include "rl.cpp"

struct ValueNetwork : Network {
    using Network::Network;
    float evaluate(Board board) const {
        float total = 0;
        for (int symmetry = 0; symmetry < 4; ++symmetry) total += value(board, symmetry);
        return total * .25f;
    }
};

struct Record { Board board; float q[4]; };

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: outcome_label BOARDS_U64 TEACHER OUTPUT ROLLOUTS SEED\n";
        return 2;
    }
    try {
        const int rollouts = std::stoi(argv[4]); const uint64_t seed = std::stoull(argv[5]);
        ValueNetwork teacher(0); teacher.load(argv[2]); Moves moves;
        std::ifstream input(argv[1], std::ios::binary); std::ofstream output(argv[3], std::ios::binary);
        Board initial; uint64_t index = 0; auto started = std::chrono::steady_clock::now();
        while (input.read(reinterpret_cast<char*>(&initial), sizeof(initial))) {
            Record record{initial, {-1e30f,-1e30f,-1e30f,-1e30f}};
            for (int first = 0; first < 4; ++first) {
                int first_reward; Board forced = moves.move(initial, first, first_reward); if (forced == initial) continue;
                int wins = 0;
                for (int rollout = 0; rollout < rollouts; ++rollout) {
                    std::mt19937_64 rng(seed + index * 37 + first * 1000000007ULL + rollout * 1000003ULL);
                    Board board = forced; bool won = rank_max(board) >= 11;
                    if (!won) board = spawn(board, rng);
                    for (int step = 0; !won && step < 3000; ++step) {
                        int mask = moves.mask(board); if (!mask) break;
                        int action = -1; float best = -1e30f;
                        for (int candidate = 0; candidate < 4; ++candidate) if (mask & (1 << candidate)) {
                            int reward; Board after = moves.move(board, candidate, reward);
                            float q = reward + teacher.evaluate(after);
                            if (rank_max(after) >= 11) q = 1e30f;
                            if (q > best) { best = q; action = candidate; }
                        }
                        int reward; Board after = moves.move(board, action, reward);
                        won = rank_max(after) >= 11; board = won ? after : spawn(after, rng);
                    }
                    wins += won;
                }
                record.q[first] = float(wins) / rollouts;
            }
            output.write(reinterpret_cast<const char*>(&record), sizeof(record)); ++index;
            if (index % 1000 == 0) {
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
                std::cerr << "{\"states\":" << index << ",\"seconds\":" << seconds << "}\n";
            }
        }
        if (!input.eof() || !output) throw std::runtime_error("outcome label I/O failure");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

// Collect current-board rollouts from a direct n-tuple policy and label with TD teacher.
#define RL_LIBRARY
#include "rl.cpp"
#include <iomanip>

struct LabelRecord {
    uint64_t board;
    float q[4];
};

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: direct_dagger POLICY TEACHER GAMES SEED [FAILURE_REPLAY]\n";
        return 2;
    }
    Network policy(0), teacher(0);
    policy.load(argv[1]);
    teacher.load(argv[2]);
    const int games = std::stoi(argv[3]);
    const uint64_t seed = std::stoull(argv[4]);
    const size_t failure_replay = argc > 5 ? std::stoull(argv[5]) : 0;
    Moves moves;
    int wins = 0;
    uint64_t states = 0;
    for (int episode = 0; episode < games; ++episode) {
        std::mt19937_64 rng(seed + episode);
        Board board = spawn(spawn(0, rng), rng);
        int largest = rank_max(board);
        std::vector<LabelRecord> trajectory;
        for (int step = 0; step < 30000 && largest < 11; ++step) {
            LabelRecord record{board, {-1e30f, -1e30f, -1e30f, -1e30f}};
            int legal_mask = 0;
            for (int action = 0; action < 4; ++action) {
                int reward;
                const Board after = moves.move(board, action, reward);
                if (after == board) continue;
                legal_mask |= 1 << action;
                float value = 0;
                for (int symmetry = 0; symmetry < 4; ++symmetry)
                    value += teacher.value(after, symmetry);
                record.q[action] = reward + value * 0.25f;
            }
            if (!legal_mask) break;
            std::cout.write(reinterpret_cast<const char*>(&record), sizeof(record));
            ++states;
            if (failure_replay) trajectory.push_back(record);
            float ignored;
            const int action = policy.choose(board, legal_mask, ignored);
            int reward;
            board = moves.move(board, action, reward);
            largest = std::max(largest, rank_max(board));
            board = spawn(board, rng);
        }
        if (largest < 11 && failure_replay) {
            const size_t begin = trajectory.size() > failure_replay
                ? trajectory.size() - failure_replay : 0;
            for (size_t index = begin; index < trajectory.size(); ++index) {
                std::cout.write(reinterpret_cast<const char*>(&trajectory[index]), sizeof(LabelRecord));
                ++states;
            }
        }
        wins += largest >= 11;
    }
    std::cerr << "{\"games\":" << games << ",\"wins\":" << wins
              << ",\"states\":" << states << "}\n";
}

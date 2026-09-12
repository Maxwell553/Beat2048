// Continue a learned score critic with terminal win/loss shaping.
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

struct Step { Board after; float value; int reward; };

int main(int argc, char** argv) {
    if (argc < 10) {
        std::cerr << "usage: win_shaped_value GAMES INPUT OUTPUT SEED ALPHA LAMBDA EPSILON WIN_BONUS SNAPSHOT_EVERY\n";
        return 2;
    }
    try {
        const int games=std::stoi(argv[1]); const std::string input=argv[2],output=argv[3]; const uint64_t seed=std::stoull(argv[4]);
        const float alpha=std::stof(argv[5]),lambda=std::stof(argv[6]),epsilon=std::stof(argv[7]),bonus=std::stof(argv[8]); const int snapshot=std::stoi(argv[9]);
        ValueNetwork network(0); network.load(input); Moves moves; auto started=std::chrono::steady_clock::now(); int wins=0,losses=0;
        for (int episode=1; episode<=games; ++episode) {
            std::mt19937_64 rng(seed+episode-1); Board board=spawn(spawn(0,rng),rng); std::vector<Step> trajectory; trajectory.reserve(2000); bool won=false;
            for (int count=0; count<30000; ++count) {
                int mask=moves.mask(board); if (!mask) break; int action=-1,best_reward=0; float best=-1e30f,best_value=0; Board best_after=board; std::vector<int> legal;
                for (int candidate=0; candidate<4; ++candidate) if (mask&(1<<candidate)) {
                    legal.push_back(candidate); int reward; Board after=moves.move(board,candidate,reward); float value=network.evaluate(after); float q=reward+value;
                    if (rank_max(after)>=11) q=1e30f;
                    if (q>best) { best=q; action=candidate; best_reward=reward; best_value=value; best_after=after; }
                }
                if (std::uniform_real_distribution<float>(0,1)(rng)<epsilon) {
                    action=legal[std::uniform_int_distribution<int>(0,int(legal.size())-1)(rng)]; best_after=moves.move(board,action,best_reward); best_value=network.evaluate(best_after);
                }
                trajectory.push_back({best_after,best_value,best_reward}); won=rank_max(best_after)>=11; if (won) break; board=spawn(best_after,rng);
            }
            float target=won?bonus:-bonus;
            for (auto it=trajectory.rbegin(); it!=trajectory.rend(); ++it) {
                network.learn(it->after,target-network.evaluate(it->after),alpha);
                target=it->reward+lambda*target+(1-lambda)*it->value;
            }
            wins+=won;losses+=!won;
            if (episode%1000==0) { double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(); std::cerr<<"{\"episodes\":"<<episode<<",\"wins\":"<<wins<<",\"losses\":"<<losses<<",\"seconds\":"<<seconds<<"}\n";wins=losses=0; }
            if (episode%snapshot==0 || episode==games) { std::string path=output; if (episode!=games) path += ".episode"+std::to_string(episode); network.save(path); }
        }
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}

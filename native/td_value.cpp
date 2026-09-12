// Training-only TD afterstate value teacher. No expectimax or handcrafted evaluation.
#define RL_LIBRARY
#include "rl.cpp"
struct ValueNetwork:Network {
    using Network::Network;
    float evaluate(Board b) const {float total=0;for(int a=0;a<4;++a)total+=value(b,a);return total*.25f;}
    void learn(Board b,float error,float alpha) {
        size_t ids[4*ACTIVE];for(int a=0;a<4;++a)indices(b,a,ids+a*ACTIVE);
        // For performance, duplicate contributions use the usual active-feature normalization.
        float step=alpha*error/ACTIVE/4;
        for(auto id:ids)weights[id]+=step;
    }
};
struct Step {Board after;float value;int reward;};
extern "C" {
void* teacher_open(const char* file){try{auto n=new ValueNetwork;n->load(file);return n;}catch(...){return nullptr;}}
void teacher_close(void* p){delete static_cast<ValueNetwork*>(p);}
void teacher_scores(void* p,const uint64_t* boards,int count,float* output){
    static Moves moves;auto n=static_cast<ValueNetwork*>(p);
    for(int i=0;i<count;++i)for(int a=0;a<4;++a){
        int reward;Board after=moves.move(boards[i],a,reward);
        output[i*4+a]=after==boards[i]?-1e30f:reward+n->evaluate(after);
    }
}
}
#ifndef VALUE_LIBRARY
int main(int argc,char** argv) {
    try {
        std::string command=argc>1?argv[1]:"train";int games=argc>2?std::stoi(argv[2]):100000;
        std::string file=argc>3?argv[3]:"models/rl/value_teacher.bin";
        int seed=argc>4?std::stoi(argv[4]):50000000;
        float alpha=argc>5?std::stof(argv[5]):.1f;
        float lambda=argc>6?std::stof(argv[6]):.5f;
        ValueNetwork network;Moves moves;if(command!="train")network.load(file);
        auto start=std::chrono::steady_clock::now();int wins=0,maxrank=0;int64_t scores=0,steps=0;
        for(int episode=1;episode<=games;++episode) {
            std::mt19937_64 rng(uint64_t(seed)+episode-1);Board b=spawn(spawn(0,rng),rng);
            std::vector<Step> trajectory;int score=0,count=0,largest=0;
            while(count<30000) {
                float best=-1e30f,value=0;int best_reward=0;Board best_after=b;float action_values[4]={-1e30f,-1e30f,-1e30f,-1e30f};
                for(int a=0;a<4;++a) {
                    int reward;Board after=moves.move(b,a,reward);if(after==b)continue;
                    float v=network.evaluate(after),q=reward+v;action_values[a]=q;
                    if(q>best){best=q;value=v;best_after=after;best_reward=reward;}
                }
                if(best_after==b)break;
                if(command=="collect"){std::cout.write(reinterpret_cast<const char*>(&b),8);std::cout.write(reinterpret_cast<const char*>(action_values),16);}
                trajectory.push_back({best_after,value,best_reward});score+=best_reward;++count;
                largest=std::max(largest,rank_max(best_after));b=spawn(best_after,rng);
                if(largest>=15 || ((command=="eval" || command=="collect") && largest>=11))break;
            }
            if(command=="train" || command=="resume") {
                float target=0;
                for(auto it=trajectory.rbegin();it!=trajectory.rend();++it) {
                    network.learn(it->after,target-network.evaluate(it->after),alpha);
                    target=it->reward+lambda*target+(1-lambda)*it->value;
                }
            }
            wins+=largest>=11;scores+=score;steps+=count;maxrank=std::max(maxrank,largest);
            if(command=="eval")std::cout<<"{\"seed\":"<<uint64_t(seed)+episode-1<<",\"won\":"<<(largest>=11?"true":"false")<<",\"score\":"<<score<<",\"moves\":"<<count<<",\"max_tile\":"<<(1<<largest)<<"}"<<std::endl;
            if(episode%1000==0 || episode==games) {
                int n=episode%1000?episode%1000:1000;double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cerr<<"{\"episodes\":"<<episode<<",\"window_games\":"<<n<<",\"wins\":"<<wins<<",\"mean_score\":"<<double(scores)/n<<",\"mean_moves\":"<<double(steps)/n<<",\"max_tile\":"<<(1<<maxrank)<<",\"seconds\":"<<seconds<<"}"<<std::endl;
                wins=0;scores=steps=0;maxrank=0;
            }
            if((command=="train" || command=="resume") && (episode%10000==0 || episode==games))network.save(file);
        }
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}

#endif

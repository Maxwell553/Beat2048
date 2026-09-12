// Direct action-value sparse neural network trained by on-policy TD(lambda).
// No search, heuristic value function, afterstate features, or spawn enumeration.
#include "rl_engine.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <stdexcept>
#if defined(WIDE_DIRECT) || defined(WIDE_DIRECT_FULL)
// Eight complementary 6-tuples share the original parameter budget through
// deterministic feature hashing. This doubles spatial coverage without making
// the checkpoint larger than the four-pattern network.
#ifdef WIDE_DIRECT_FULL
static constexpr size_t WIDTH=1<<24;
static constexpr uint64_t HEADER_MAGIC=0x2048524c51303033ULL;
#else
static constexpr size_t WIDTH=1<<23;
static constexpr uint64_t HEADER_MAGIC=0x2048524c51303032ULL;
#endif
static constexpr int PATTERNS=8,ACTIVE=16;
static const int patterns[PATTERNS][6]={
    {0,1,2,4,5,6},{4,5,6,8,9,10},{0,1,2,3,4,5},{4,5,6,7,8,9},
    {0,1,4,5,8,9},{1,2,5,6,9,10},{0,1,2,5,6,7},{4,5,6,9,10,11}
};
#else
static constexpr size_t WIDTH=1<<24;
static constexpr int PATTERNS=4,ACTIVE=8;
static constexpr uint64_t HEADER_MAGIC=0x2048524c51303031ULL;
static const int patterns[PATTERNS][6]={{0,1,2,4,5,6},{4,5,6,8,9,10},{0,1,2,3,4,5},{4,5,6,7,8,9}};
#endif
static constexpr size_t PARAMETERS=PATTERNS*WIDTH;
struct Network {
    std::vector<float> weights;
    int cells[4][2][PATTERNS][6];
    Network(float optimistic=100000):weights(PARAMETERS,optimistic/ACTIVE) {
        for(int action=0;action<4;++action)for(int reflect=0;reflect<2;++reflect)
            for(int p=0;p<PATTERNS;++p)for(int j=0;j<6;++j) {
                int r=patterns[p][j]/4,c=patterns[p][j]%4;
                if(reflect)r=3-r;
                int rr=r,cc=c;
                // Map a board rotated into left-moving coordinates back to its input.
                if(action==0){rr=c;cc=3-r;}
                if(action==1){rr=3-r;cc=3-c;}
                if(action==2){rr=3-c;cc=r;}
                cells[action][reflect][p][j]=4*rr+cc;
            }
    }
    void indices(Board b,int action,size_t* ids) const {
        int k=0;
        for(int reflect=0;reflect<2;++reflect)for(int p=0;p<PATTERNS;++p) {
            unsigned index=0;
            for(int j=0;j<6;++j)index|=unsigned((b>>(4*cells[action][reflect][p][j]))&15)<<(4*j);
            ids[k++]=p*WIDTH+(index&(WIDTH-1));
        }
    }
    float value(Board b,int action) const {
        size_t ids[ACTIVE];indices(b,action,ids);float value=0;
        for(auto id:ids)value+=weights[id];
        return value;
    }
    int choose(Board b,int mask,float& best) const {
        int action=-1;best=-1e30f;
        for(int a=0;a<4;++a)if(mask&(1<<a)) {
            float q=value(b,a);if(q>best){best=q;action=a;}
        }
        return action;
    }
    void update(Board b,int action,float error,float alpha) {
        size_t ids[ACTIVE];indices(b,action,ids);
        // Normalize the sparse gradient, accounting for repeated active indices.
        int norm=0;
        for(int i=0;i<ACTIVE;++i)for(int j=0;j<ACTIVE;++j)norm+=ids[i]==ids[j];
        float step=alpha*error/norm;
        for(auto id:ids)weights[id]+=step;
    }
    void save(const std::string& file) const {
        std::string temp=file+".tmp";
        std::ofstream f(temp,std::ios::binary);uint64_t header[3]={HEADER_MAGIC,PARAMETERS,1};
        f.write(reinterpret_cast<const char*>(header),sizeof(header));
        f.write(reinterpret_cast<const char*>(weights.data()),weights.size()*sizeof(float));f.close();
        if(!f)throw std::runtime_error("checkpoint write failed");
        if(std::rename(temp.c_str(),file.c_str()))throw std::runtime_error("checkpoint rename failed");
    }
    void load(const std::string& file) {
        std::ifstream f(file,std::ios::binary);uint64_t header[3]={0};f.read(reinterpret_cast<char*>(header),sizeof(header));
        if(header[0]!=HEADER_MAGIC || header[1]!=PARAMETERS)throw std::runtime_error("wrong checkpoint format");
        f.read(reinterpret_cast<char*>(weights.data()),weights.size()*sizeof(float));
        if(!f)throw std::runtime_error("truncated checkpoint");
    }
};
extern "C" {
void* rl_open(const char* file){try{auto n=new Network(0);n->load(file);return n;}catch(...){return nullptr;}}
void rl_close(void* pointer){delete static_cast<Network*>(pointer);}
void rl_values(void* pointer,uint64_t board,float* output){auto n=static_cast<Network*>(pointer);for(int a=0;a<4;++a)output[a]=n->value(board,a);}
void rl_values_batch(void* pointer,const uint64_t* boards,int count,float* output){
    auto n=static_cast<Network*>(pointer);
    for(int i=0;i<count;++i)for(int a=0;a<4;++a)output[i*4+a]=n->value(boards[i],a);
}
uint64_t rl_move(uint64_t board,int action,int* reward){static Moves moves;return moves.move(board,action,*reward);}
void rl_move_batch(const uint64_t* boards,int count,int action,uint64_t* output,int* rewards){
    static Moves moves;
    for(int i=0;i<count;++i)output[i]=moves.move(boards[i],action,rewards[i]);
}
}
#ifndef RL_LIBRARY
struct Experience {Board board;float q;int action,reward;};
int main(int argc,char** argv) {
    try {
        std::string command=argc>1?argv[1]:"train";
        int games=argc>2?std::stoi(argv[2]):100000;
        std::string file=argc>3?argv[3]:"models/rl/direct_q.bin";
        int seed=argc>4?std::stoi(argv[4]):20260909;
        float alpha=argc>5?std::stof(argv[5]):.1f;
        float lambda=argc>6?std::stof(argv[6]):.5f;
        Network network;Moves moves;
        if(command=="eval" || command=="resume")network.load(file);
        if(command=="train")network.save(file+".initial");
        auto start=std::chrono::steady_clock::now();
        int wins=0;int64_t scores=0,steps=0;int maxrank=0;
        for(int episode=1;episode<=games;++episode) {
            std::mt19937_64 rng(uint64_t(seed)+episode-1);
            Board b=spawn(spawn(0,rng),rng);int score=0,count=0,largest=0;
            std::vector<Experience> trajectory;trajectory.reserve(5000);
            for(;count<30000;++count) {
                int mask=moves.mask(b);if(!mask)break;
                float q;int action=network.choose(b,mask,q);
                int reward;Board after=moves.move(b,action,reward);
                trajectory.push_back({b,q,action,reward});
                score+=reward;largest=std::max(largest,rank_max(after));
                b=spawn(after,rng);
                if(largest>=15 || (command=="eval" && largest>=11)){++count;break;}
            }
            if(command!="eval") {
                float target=0,next_q=0;
                for(auto it=trajectory.rbegin();it!=trajectory.rend();++it) {
                    target=it->reward+lambda*target+(1-lambda)*next_q;
                    float current=network.value(it->board,it->action);
                    network.update(it->board,it->action,target-current,alpha);
                    next_q=it->q;
                }
            }
            wins+=largest>=11;scores+=score;steps+=count;maxrank=std::max(maxrank,largest);
            if(command=="eval")std::cout<<"{\"seed\":"<<uint64_t(seed)+episode-1<<",\"won\":"<<(largest>=11?"true":"false")<<",\"score\":"<<score<<",\"moves\":"<<count<<",\"max_tile\":"<<(1<<largest)<<"}"<<std::endl;
            if(episode%1000==0 || episode==games) {
                int divisor=episode%1000?episode%1000:1000;
                double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cerr<<"{\"episodes\":"<<episode<<",\"window_games\":"<<divisor<<",\"wins\":"<<wins<<",\"mean_score\":"<<double(scores)/divisor<<",\"mean_moves\":"<<double(steps)/divisor<<",\"max_tile\":"<<(1<<maxrank)<<",\"seconds\":"<<seconds<<"}"<<std::endl;
                wins=0;scores=steps=0;maxrank=0;
            }
            if(command!="eval" && (episode%10000==0 || episode==games))network.save(file);
        }
    } catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
#endif

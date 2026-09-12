// Batched real game transitions for neural-policy RL. No policy evaluation or search.
#include "rl_engine.h"
#include <vector>
struct Batch {
    Moves moves;std::vector<Board> boards;std::vector<std::mt19937_64> rng;
    int wins=0,losses=0;std::vector<Board> curriculum;double curriculum_probability=0;
    Board reset(size_t i){
        if(!curriculum.empty() && std::uniform_real_distribution<double>(0,1)(rng[i])<curriculum_probability)
            return curriculum[std::uniform_int_distribution<size_t>(0,curriculum.size()-1)(rng[i])];
        return spawn(spawn(0,rng[i]),rng[i]);
    }
    Batch(int n,uint64_t seed):boards(n,0) {
        for(int i=0;i<n;++i){rng.emplace_back(seed+i);boards[i]=spawn(spawn(0,rng[i]),rng[i]);}
    }
    void states(uint8_t* output,uint8_t* legal) {
        for(size_t i=0;i<boards.size();++i) {
            for(int j=0;j<16;++j)output[16*i+j]=(boards[i]>>(4*j))&15;
            int mask=moves.mask(boards[i]);for(int a=0;a<4;++a)legal[4*i+a]=(mask>>a)&1;
        }
    }
    void set_boards(const Board* input) {
        std::copy(input,input+boards.size(),boards.begin());
    }
    void step(const int32_t* actions,float* rewards,float* terminal,uint8_t* output,uint8_t* legal,int32_t* outcomes) {
        for(size_t i=0;i<boards.size();++i) {
            int reward=0;Board next=moves.move(boards[i],actions[i],reward);
            bool valid=next!=boards[i];
            if(valid)next=spawn(next,rng[i]);
            bool won=rank_max(next)>=11,lost=moves.mask(next)==0;
            rewards[i]=reward/2048.f+(won?10.f:0.f)-(lost?1.f:0.f);terminal[i]=won||lost;
            outcomes[i]=won?1:(lost?-1:0);
            if(won||lost){wins+=won;losses+=lost;next=reset(i);}
            boards[i]=next;
        }
        states(output,legal);
    }
};
extern "C" {
void* batch_open(int n,uint64_t seed){return new Batch(n,seed);}
void batch_curriculum(void* p,const uint64_t* boards,int n,double probability){auto b=static_cast<Batch*>(p);b->curriculum.assign(boards,boards+n);b->curriculum_probability=probability;}
void batch_close(void* p){delete static_cast<Batch*>(p);}
void batch_states(void* p,uint8_t* boards,uint8_t* legal){static_cast<Batch*>(p)->states(boards,legal);}
void batch_set_boards(void* p,const uint64_t* boards){static_cast<Batch*>(p)->set_boards(boards);}
void batch_step(void* p,const int32_t* a,float* r,float* d,uint8_t* b,uint8_t* l,int32_t* o){static_cast<Batch*>(p)->step(a,r,d,b,l,o);}
}

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
using Board=uint64_t;
struct Moves {
    uint16_t left[65536],right[65536];
    int rewards[65536];
    Moves() {
        for(int row=0;row<65536;++row) {
            int a[4],n=0,out[4]={0},k=0,score=0;
            for(int i=0;i<4;++i) {int v=(row>>(i*4))&15;if(v)a[n++]=v;}
            for(int i=0;i<n;++i) {
                if(i+1<n && a[i]==a[i+1]) {out[k]=std::min(15,a[i]+1);score+=1<<out[k++];++i;}
                else out[k++]=a[i];
            }
            uint16_t result=0;
            for(int i=0;i<4;++i)result|=out[i]<<(4*i);
            left[row]=result;right[reverse(row)]=reverse(result);rewards[row]=score;
        }
    }
    static uint16_t reverse(uint16_t r) {return ((r&15)<<12)|((r&240)<<4)|((r&3840)>>4)|((r&61440)>>12);}
    static Board transpose(Board b) {
        // Bit permutation adapted from nneonneo/2048-ai (MIT; see THIRD_PARTY_NOTICES.md).
        Board a=(b&0xF0F00F0FF0F00F0FULL)|((b&0x0000F0F00000F0F0ULL)<<12)|((b&0x0F0F00000F0F0000ULL)>>12);
        return (a&0xFF00FF0000FF00FFULL)|((a&0x00FF00FF00000000ULL)>>24)|((a&0x00000000FF00FF00ULL)<<24);
    }
    Board move(Board b,int action,int& reward) const {
        bool vertical=action==0 || action==2;
        if(vertical)b=transpose(b);
        Board result=0;reward=0;
        for(int i=0;i<4;++i) {
            uint16_t row=(b>>(16*i))&65535;
            result|=Board((action==0 || action==3)?left[row]:right[row])<<(16*i);
            reward+=rewards[row];
        }
        return vertical?transpose(result):result;
    }
    int mask(Board b) const {
        int mask=0,reward;
        for(int a=0;a<4;++a)if(move(b,a,reward)!=b)mask|=1<<a;
        return mask;
    }
};
inline int rank_max(Board b) {int m=0;for(int i=0;i<16;++i)m=std::max(m,int((b>>(4*i))&15));return m;}
inline Board spawn(Board b,std::mt19937_64& rng) {
    int empty[16],n=0;
    for(int i=0;i<16;++i)if(!((b>>(4*i))&15))empty[n++]=i;
    if(!n)return b;
    int cell=empty[std::uniform_int_distribution<int>(0,n-1)(rng)];
    int rank=std::uniform_real_distribution<double>(0,1)(rng)<.9?1:2;
    return b|(Board(rank)<<(4*cell));
}

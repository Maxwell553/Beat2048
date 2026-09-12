// Row heuristic and bitwise transpose adapted from nneonneo/2048-ai.
// Copyright (c) 2014-2019 Robert Xiao and contributors. MIT; see THIRD_PARTY_NOTICES.md.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>

using Board = uint64_t;
static uint16_t left_rows[65536], right_rows[65536];
static double heuristics[65536];
static std::once_flag initialized;
static uint16_t reverse_row(uint16_t r) {
    return ((r & 0xf) << 12) | ((r & 0xf0) << 4) | ((r & 0xf00) >> 4) | ((r & 0xf000) >> 12);
}
static void init() {
    for (unsigned r = 0; r < 65536; ++r) {
        int a[4], compact[4], n=0, out[4]={0}, k=0;
        double sum=0, mono_l=0, mono_r=0;
        int empty=0, merges=0, previous=0, run=0;
        for (int i=0;i<4;++i) {
            a[i]=(r>>(4*i))&15;
            if (a[i]) compact[n++]=a[i]; else ++empty;
            sum += std::pow(a[i], 3.5);
            if (a[i]) {
                if (previous==a[i]) ++run;
                else { if (run) merges += 1+run; run=0; }
                previous=a[i];
            }
        }
        if (run) merges+=1+run;
        for(int i=0;i<3;++i) {
            double d=std::pow(a[i],4)-std::pow(a[i+1],4);
            if(d>0) mono_l+=d; else mono_r-=d;
        }
        heuristics[r]=200000.0 + 270.0*empty + 700.0*merges - 47.0*std::min(mono_l,mono_r) - 11.0*sum;
        for(int i=0;i<n;++i) {
            if(i+1<n && compact[i]==compact[i+1]) {out[k++]=std::min(15,compact[i]+1);++i;}
            else out[k++]=compact[i];
        }
        uint16_t moved=0;
        for(int i=0;i<4;++i) moved |= out[i]<<(4*i);
        left_rows[r]=moved;
        right_rows[reverse_row(r)]=reverse_row(moved);
    }
}
static Board transpose(Board b) {
    Board a1=b & 0xF0F00F0FF0F00F0FULL;
    Board a2=b & 0x0000F0F00000F0F0ULL;
    Board a3=b & 0x0F0F00000F0F0000ULL;
    Board a=a1 | (a2<<12) | (a3>>12);
    return (a & 0xFF00FF0000FF00FFULL) |
           ((a & 0x00FF00FF00000000ULL)>>24) |
           ((a & 0x00000000FF00FF00ULL)<<24);
}
// 0 up, 1 right, 2 down, 3 left; identical to Python environment.
static Board move_board(Board b,int action) {
    bool vertical=(action==0 || action==2);
    if(vertical) b=transpose(b);
    Board out=0;
    auto table=(action==0 || action==3)?left_rows:right_rows;
    for(int r=0;r<4;++r) out |= Board(table[(b>>(16*r))&65535])<<(16*r);
    return vertical?transpose(out):out;
}
static double heuristic(Board b) {
    Board t=transpose(b); double h=0;
    for(int r=0;r<4;++r) h += heuristics[(b>>(16*r))&65535]+heuristics[(t>>(16*r))&65535];
    return h;
}
struct Search {
    double cutoff;
    uint64_t nodes=0;
    double player(Board b,int depth,double probability) {
        ++nodes;
        if(depth<=0 || probability<cutoff) return heuristic(b);
        double best=-1e100;
        for(int a=0;a<4;++a) {Board next=move_board(b,a); if(next!=b) best=std::max(best,chance(next,depth-1,probability));}
        return best == -1e100 ? 0.0 : best;
    }
    double chance(Board b,int depth,double probability) {
        ++nodes;
        if(depth<=0 || probability<cutoff) return heuristic(b);
        int cells[16], n=0;
        for(int i=0;i<16;++i) if(!((b>>(4*i))&15)) cells[n++]=i;
        if(!n) return player(b,depth,probability);
        double total=0;
        for(int j=0;j<n;++j) {
            int shift=4*cells[j];
            total += .9*player(b | (Board(1)<<shift),depth,probability*.9/n);
            total += .1*player(b | (Board(2)<<shift),depth,probability*.1/n);
        }
        return total/n;
    }
};
extern "C" {
void search_scores(uint64_t board,int depth,double cutoff,double* scores,uint64_t* nodes) {
    std::call_once(initialized,init);
    Search search{cutoff};
    for(int a=0;a<4;++a) {
        Board next=move_board(board,a);
        scores[a]=next==board ? -1e100 : search.chance(next,depth,1.0);
    }
    *nodes=search.nodes;
}
uint64_t native_move(uint64_t board,int action) {std::call_once(initialized,init);return move_board(board,action);}
}

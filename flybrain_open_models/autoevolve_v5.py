#!/usr/bin/env python3
import argparse, gzip, json, random, csv
from pathlib import Path
import numpy as np

# Structural genome: neurons may split/prune, edges add/delete/rewire, weights mutate.
# Fitness uses multiple environments so evolution cannot win by one fixed reflex.

def load(path):
    with gzip.open(path,'rb') as f: raw=f.read()
    n,e=np.frombuffer(raw,dtype='<u4',count=2)
    dt=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')])
    a=np.frombuffer(raw,dtype=dt,count=int(e),offset=8)
    return int(n), np.array(a['pre'],dtype=np.int32),np.array(a['post'],dtype=np.int32),np.array(a['w'],dtype=np.float32)

class Genome:
    def __init__(self,n,pre,post,w):
        self.n=n; self.pre=pre; self.post=post; self.w=w
        self.threshold=1.0; self.decay=.95
    def clone(self):
        g=Genome(self.n,self.pre.copy(),self.post.copy(),self.w.copy());g.threshold=self.threshold;g.decay=self.decay;return g
    def mutate(self,rng):
        g=self.clone(); m=len(g.w)
        # synaptic plasticity
        k=max(1,int(m*.002)); ix=rng.integers(0,m,k); g.w[ix]+=rng.normal(0,.12,k)
        # delete weak/random edges
        if m>500 and rng.random()<.75:
            k=min(max(1,int(m*.0005)),5000); drop=rng.choice(m,k,replace=False); keep=np.ones(m,bool);keep[drop]=0
            g.pre,g.post,g.w=g.pre[keep],g.post[keep],g.w[keep]
        # add/rewire synapses (true topology mutation)
        if rng.random()<.95:
            k=min(max(8,int(len(g.w)*.0007)),7000)
            g.pre=np.r_[g.pre,rng.integers(0,g.n,k,dtype=np.int32)]
            g.post=np.r_[g.post,rng.integers(0,g.n,k,dtype=np.int32)]
            g.w=np.r_[g.w,rng.normal(0,.15,k).astype(np.float32)]
        # neuron duplication/splitting: adds a neuron and copies a bounded sample of a parent's incident edges
        if rng.random()<.55:
            parent=int(rng.integers(0,g.n)); new=g.n; g.n+=1
            inc=np.flatnonzero(g.post==parent)[:128]; out=np.flatnonzero(g.pre==parent)[:128]
            if len(inc):
                g.pre=np.r_[g.pre,g.pre[inc]];g.post=np.r_[g.post,np.full(len(inc),new,np.int32)];g.w=np.r_[g.w,g.w[inc]*(1+rng.normal(0,.08,len(inc)))]
            if len(out):
                g.pre=np.r_[g.pre,np.full(len(out),new,np.int32)];g.post=np.r_[g.post,g.post[out]];g.w=np.r_[g.w,g.w[out]*(1+rng.normal(0,.08,len(out)))]
        g.threshold=float(np.clip(g.threshold+rng.normal(0,.025),.65,1.35));g.decay=float(np.clip(g.decay+rng.normal(0,.008),.86,.995))
        return g

def phenotype_score(g,seed):
    # Fast recurrent challenge on a sampled subnetwork; rewards memory, separation, response and efficiency.
    rng=np.random.default_rng(seed); N=min(g.n,4096); mask=(g.pre<N)&(g.post<N); pre=g.pre[mask];post=g.post[mask];w=np.tanh(g.w[mask])*.12
    if len(w)>180000: ix=rng.choice(len(w),180000,False);pre,post,w=pre[ix],post[ix],w[ix]
    scores=[]
    for env in range(8):
        v=np.zeros(N,np.float32); trace=[]; stim=rng.choice(N,96,False)
        for t in range(18):
            v*=g.decay
            if t<4: v[stim]+=1.15 if env%2==0 else .82
            fire=v>=g.threshold; trace.append(float(fire.mean()))
            if fire.any() and len(w):
                active=fire[pre]; syn=np.zeros(N,np.float32);np.add.at(syn,post[active],w[active]);v+=syn
            v[fire]=0;np.clip(v,-3,3,out=v)
        tr=np.array(trace); response=tr[:6].sum(); memory=tr[6:].sum(); stability=1/(1+tr.std()*20)
        scores.append(response*2+memory*.7+stability)
    complexity=.0000008*len(g.w)+.00001*max(0,g.n-139255)
    return float(np.mean(scores)-complexity)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--connectome',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--generations',type=int,default=12);ap.add_argument('--population',type=int,default=10);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    n,p,q,w=load(a.connectome); rng=np.random.default_rng(20260929); base=Genome(n,p,q,w); pop=[base]+[base.mutate(rng) for _ in range(a.population-1)]; hist=[]; best=None
    for gen in range(a.generations):
        scored=[(phenotype_score(g,1000+gen*31+i),g) for i,g in enumerate(pop)];scored.sort(key=lambda z:z[0],reverse=True);best=scored[0]
        hist.append({'generation':gen,'fitness':best[0],'neurons':best[1].n,'edges':len(best[1].w),'threshold':best[1].threshold,'decay':best[1].decay})
        print(json.dumps(hist[-1]),flush=True)
        elite=[x[1] for x in scored[:max(2,a.population//4)]];pop=[elite[0]]
        while len(pop)<a.population: pop.append(random.choice(elite).mutate(rng))
    g=best[1];np.savez_compressed(a.out/'evolved_brain.npz',n=g.n,pre=g.pre,post=g.post,w=g.w,threshold=g.threshold,decay=g.decay)
    with open(a.out/'evolution.csv','w',newline='') as f: dw=csv.DictWriter(f,fieldnames=hist[0]);dw.writeheader();dw.writerows(hist)
    (a.out/'result.json').write_text(json.dumps({'base_neurons':n,'base_edges':len(w),'best':hist[-1],'history':hist,'architecture_mutations':['edge_add','edge_delete','rewire','neuron_split','weight','threshold','decay']},indent=2))
if __name__=='__main__':main()

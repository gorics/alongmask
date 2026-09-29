from __future__ import annotations
import gzip, json, struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT=Path('flyevo'); OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'flybrain-connectome.bin.gz'
if not DATA.exists():
    raise SystemExit('Run full_connectome.py first so the packaged full connectome is present.')
raw=gzip.decompress(DATA.read_bytes())
n,e=struct.unpack_from('<II',raw,0)
edge_dtype=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')])
edges=np.frombuffer(raw,dtype=edge_dtype,count=e,offset=8)
off=8+e*12
node_dtype=np.dtype([('region','u1'),('group','<u2')])
nodes=np.frombuffer(raw,dtype=node_dtype,count=n,offset=off)
pre=edges['pre'].astype(np.int64); post=edges['post'].astype(np.int64); ew=edges['w'].astype(np.float32)
W=sp.csr_matrix((ew,(pre,post)),shape=(n,n),dtype=np.float32); W.sum_duplicates()
sens=np.flatnonzero(nodes['region']==0); central=np.flatnonzero(nodes['region']==1); motor=np.flatnonzero(nodes['region']==3)
print(f'full graph: n={n}, e={e}, sensory={len(sens)}, central={len(central)}, motor={len(motor)}')

@dataclass
class Patch:
    add_pre: np.ndarray
    add_post: np.ndarray
    add_w: np.ndarray
    del_idx: np.ndarray
    mutation_rate: float=0.10

    @staticmethod
    def empty():
        return Patch(np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0,np.float32),np.empty(0,np.int64),0.10)
    def clone(self):
        return Patch(self.add_pre.copy(),self.add_post.copy(),self.add_w.copy(),self.del_idx.copy(),self.mutation_rate)
    def mutate(self,rng):
        q=self.clone()
        # structural births: new long-range edges become part of this full-brain individual
        births=max(1,int(rng.poisson(3)))
        src_pool=np.concatenate([sens,central])
        dst_pool=np.concatenate([central,motor])
        q.add_pre=np.concatenate([q.add_pre,rng.choice(src_pool,births).astype(np.int64)])
        q.add_post=np.concatenate([q.add_post,rng.choice(dst_pool,births).astype(np.int64)])
        q.add_w=np.concatenate([q.add_w,rng.normal(18,8,births).clip(-30,35).astype(np.float32)])
        # structural deaths: remove real base-connectome edges in this individual's graph
        deaths=max(1,int(rng.poisson(2)))
        q.del_idx=np.unique(np.concatenate([q.del_idx,rng.integers(0,e,deaths,dtype=np.int64)]))
        # point mutations of added synaptic strengths
        if len(q.add_w):
            m=rng.random(len(q.add_w))<0.35
            q.add_w[m]+=rng.normal(0,5,m.sum()).astype(np.float32)
            q.add_w=np.clip(q.add_w,-40,40)
        # occasional deletion of a newly born edge
        if len(q.add_w)>2 and rng.random()<0.4:
            keep=np.ones(len(q.add_w),bool); keep[int(rng.integers(0,len(q.add_w)))]=False
            q.add_pre,q.add_post,q.add_w=q.add_pre[keep],q.add_post[keep],q.add_w[keep]
        q.mutation_rate=float(np.clip(q.mutation_rate+rng.normal(0,.01),.02,.3))
        return q


def patch_matrix(g:Patch):
    pp=[]; qq=[]; ww=[]
    if len(g.add_w):
        pp.extend(g.add_pre.tolist()); qq.extend(g.add_post.tolist()); ww.extend(g.add_w.tolist())
    if len(g.del_idx):
        ii=g.del_idx
        pp.extend(pre[ii].tolist()); qq.extend(post[ii].tolist()); ww.extend((-ew[ii]).tolist())
    if not ww: return None
    P=sp.csr_matrix((np.asarray(ww,np.float32),(np.asarray(pp),np.asarray(qq))),shape=(n,n),dtype=np.float32); P.sum_duplicates(); return P


def simulate(g:Patch,seed:int,steps=140):
    rng=np.random.default_rng(seed); P=patch_matrix(g)
    stim=rng.choice(sens,min(384,len(sens)),replace=False)
    v=np.full(n,-52.,np.float32); conduct=np.zeros(n,np.float32); refr=np.zeros(n,np.int32); prev=np.empty(0,np.int64)
    rest=-52.; reset=-52.; th=-45.; dt=.1; md=np.float32(np.exp(-dt/20)); sd=np.float32(np.exp(-dt/5)); gain=np.float32(1-md); wscale=np.float32(.275)
    motor_sp=0; internal=0; all_sp=0
    for step in range(steps):
        if prev.size:
            cur=np.asarray(W[prev].sum(axis=0)).ravel().astype(np.float32)
            if P is not None: cur+=np.asarray(P[prev].sum(axis=0)).ravel().astype(np.float32)
            conduct+=cur*wscale
        active=refr<=step; v=np.where(active,rest+(v-rest)*md+conduct*gain,v); conduct*=sd
        fired=np.flatnonzero((v>th)&active)
        ext=stim[rng.random(len(stim))<(130*dt/1000)]
        spk=np.unique(np.concatenate([fired,ext])).astype(np.int64)
        if spk.size: v[spk]=reset; conduct[spk]=0; refr[spk]=step+22
        motor_sp+=int(np.isin(fired,motor,assume_unique=False).sum())
        internal+=int(fired.size); all_sp+=int(spk.size); prev=spk
    complexity=len(g.add_w)+len(g.del_idx)
    # selection pressure: propagate sensory activity into actual motor-region neurons without global runaway firing
    fitness=4.0*motor_sp + .0015*internal - .0005*all_sp - .025*complexity
    return float(fitness),{'motor_spikes':motor_sp,'internal_spikes':internal,'all_spikes':all_sp,'added_edges':len(g.add_w),'deleted_edges':len(g.del_idx)}


def evaluate(g):
    vals=[]; details=[]
    for s in (101,202):
        f,d=simulate(g,s); vals.append(f); details.append(d)
    return float(np.mean(vals)),details

rng=np.random.default_rng(20260929)
pop=[Patch.empty()]
while len(pop)<8: pop.append(Patch.empty().mutate(rng))
hist=[]
for gen in range(8):
    scored=[]
    for g in pop:
        f,d=evaluate(g); scored.append((f,g,d))
    scored.sort(key=lambda x:x[0],reverse=True)
    bf,bg,bd=scored[0]
    row={'generation':gen,'best_fitness':bf,'mean_fitness':float(np.mean([x[0] for x in scored])),'added_edges':len(bg.add_w),'deleted_edges':len(bg.del_idx),'trial_details':bd}
    hist.append(row); print(json.dumps(row))
    elite=[x[1] for x in scored[:3]]
    pop=[elite[0].clone(),elite[1].clone()]
    while len(pop)<8: pop.append(elite[int(rng.integers(0,len(elite)))].mutate(rng))

scored=[]
for g in pop:
    f,d=evaluate(g); scored.append((f,g,d))
scored.sort(key=lambda x:x[0],reverse=True); final_f,best,final_d=scored[0]
report={
 'neurons':int(n),'base_edges':int(e),'generations':len(hist),'population':8,
 'start_best_fitness':hist[0]['best_fitness'],'last_recorded_best_fitness':hist[-1]['best_fitness'],'final_best_fitness':final_f,
 'best_added_edges':int(len(best.add_w)),'best_deleted_base_edges':int(len(best.del_idx)),
 'added_edge_examples':[{'pre':int(a),'post':int(b),'weight':float(w)} for a,b,w in zip(best.add_pre[:12],best.add_post[:12],best.add_w[:12])],
 'deleted_edge_examples':[{'edge_index':int(i),'pre':int(pre[i]),'post':int(post[i]),'weight_removed':float(ew[i])} for i in best.del_idx[:12]],
 'final_trial_details':final_d,'history':hist,
 'claim_boundary':'The underlying full 139k-node graph is retained, while each evolutionary individual has real structural edge births/deaths applied as a sparse patch during propagation. This proves graph topology evolution on the full connectome model; it does not prove unbounded intelligence.'
}
(OUT/'full_connectome_evolution.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
np.savez_compressed(OUT/'full_connectome_evolved_patch.npz',add_pre=best.add_pre,add_post=best.add_post,add_w=best.add_w,del_idx=best.del_idx)
print('FINAL',json.dumps(report))

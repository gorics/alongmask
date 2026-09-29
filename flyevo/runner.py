import json, math
from pathlib import Path
import numpy as np

OUT=Path('flyevo/results'); OUT.mkdir(parents=True,exist_ok=True)
N=64; S=12; M=6; MS=N-M

class G:
    def __init__(self,rng):
        self.w=(rng.normal(0,.45,(N,N))*(rng.random((N,N))<.06)).astype(np.float32); np.fill_diagonal(self.w,0)
        self.th=1.0; self.leak=.88; self.llm=.2; self.sd=.2; self.mut=.04
    def copy(self):
        z=object.__new__(G); z.w=self.w.copy(); z.th=self.th; z.leak=self.leak; z.llm=self.llm; z.sd=self.sd; z.mut=self.mut; return z
    def mutate(self,rng):
        z=self.copy(); q=rng.random(z.w.shape)<z.mut; z.w[q]+=rng.normal(0,.2,q.sum());
        born=(rng.random(z.w.shape)<z.mut*.12)&(np.abs(z.w)<1e-8); dead=(rng.random(z.w.shape)<z.mut*.08)&(np.abs(z.w)>0)
        z.w[born]=rng.normal(0,.35,born.sum()); z.w[dead]=0; np.fill_diagonal(z.w,0); z.w=np.clip(z.w,-2,2).astype(np.float32)
        z.th=float(np.clip(z.th+rng.normal(0,.04),.6,1.7)); z.leak=float(np.clip(z.leak+rng.normal(0,.02),.72,.98)); z.llm=float(np.clip(z.llm+rng.normal(0,.04),0,1)); z.sd=float(np.clip(z.sd+rng.normal(0,.04),0,1)); z.mut=float(np.clip(z.mut+rng.normal(0,.004),.005,.12)); return z

def run(g,seed,steps=80,capture=False):
    rng=np.random.default_rng(seed); v=np.zeros(N,np.float32); sp=np.zeros(N,np.float32); tr=np.zeros(N,np.float32)
    x=y=0.; h=float(rng.uniform(-math.pi,math.pi)); tx,ty=map(float,rng.uniform(-4,4,2)); d0=math.hypot(tx,ty); total=0.; rows=[]
    for t in range(steps):
        a=math.atan2(ty-y,tx-x)-h; a=math.atan2(math.sin(a),math.cos(a)); d=math.hypot(tx-x,ty-y); vis=not(30<=t<45)
        s=np.zeros(S,np.float32)
        if vis:
            c=np.linspace(-math.pi,math.pi,8,endpoint=False); da=np.angle(np.exp(1j*(a-c))); s[:8]=np.exp(-(da*da)/(2*.6*.6))*max(.15,1.2-d/6)
        s[8]=float(vis); s[9]=math.sin(t*.2); s[10]=math.cos(t*.2); s[11]=1
        inp=np.zeros(N,np.float32); inp[:S]=s; inp[S:S+4]+=g.llm*np.array([math.sin(t*.07+i) for i in range(4)]); inp[S+4:S+8]+=g.sd*np.array([math.cos(t*.05+i) for i in range(4)])
        v=g.leak*v+inp+sp@g.w; sp=(v>g.th).astype(np.float32); v[sp>0]=0; tr=.92*tr+sp
        turn=float(np.tanh((tr[MS]+tr[MS+1])-(tr[MS+2]+tr[MS+3]))); fwd=float(1/(1+np.exp(-(tr[MS+4]-tr[MS+5])))); h+=.22*turn; speed=.04+.1*fwd; x+=math.cos(h)*speed; y+=math.sin(h)*speed
        nd=math.hypot(tx-x,ty-y); reward=(d-nd)*3-.0015*sp.sum(); total+=reward
        if capture: rows.append({'t':t,'distance':float(nd),'visible':vis,'spikes':int(sp.sum()),'turn':turn,'state_ko':('시야 차단 · 방향 기억 유지' if not vis else '목표 방향으로 접근/회전 중')})
    fit=total+(d0-math.hypot(tx-x,ty-y))*1.5-.00002*np.count_nonzero(g.w)
    return float(fit),rows

def score(g): return float(np.mean([run(g,s)[0] for s in (11,22,33)]))

def main():
    rng=np.random.default_rng(20260929); pop=[G(rng) for _ in range(14)]; hist=[]
    for k in range(14):
        sc=sorted([(score(g),g) for g in pop],key=lambda x:x[0],reverse=True); best=sc[0][1]
        hist.append({'generation':k,'best_fitness':sc[0][0],'mean_fitness':float(np.mean([x[0] for x in sc])),'edges':int(np.count_nonzero(best.w)),'llm_gain':best.llm,'sd_gain':best.sd}); print(hist[-1])
        elite=[g for _,g in sc[:4]]; pop=[elite[0].copy(),elite[1].copy()]+[elite[int(rng.integers(0,len(elite)))].mutate(rng) for _ in range(12)]
    best=max(pop,key=score); fit,rows=run(best,777,100,True)
    np.savez_compressed(OUT/'best_genome.npz',weights=best.w,threshold=best.th,leak=best.leak,llm_gain=best.llm,sd_gain=best.sd)
    (OUT/'evolution.json').write_text(json.dumps(hist,indent=2),encoding='utf-8'); (OUT/'thought_stream.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows),encoding='utf-8')
    summary={'start_best':hist[0]['best_fitness'],'final_best':hist[-1]['best_fitness'],'improvement':hist[-1]['best_fitness']-hist[0]['best_fitness'],'edges_start':hist[0]['edges'],'edges_final':hist[-1]['edges'],'demo_fitness':fit,'thought_note':'state_ko is a simulation-state interpreter, not subjective thought'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print('SUMMARY',json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()

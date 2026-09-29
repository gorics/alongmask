#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
import numpy as np

BASE_N=139255; BASE_E=2698236
REG=np.array([103847.,35252.,80.,76.]); RN=['sensory','central','drives','motor']
TN=['thermal','desiccation','resource_breadth','pathogen_defense','dispersal','fecundity','development_speed','sociality','body_size_log','metabolic_efficiency']

def env(y):
    twopi=2*math.pi
    return dict(
      temperature=.70*math.sin(twopi*y/1.7e9)+.30*math.sin(twopi*y/13.1e9+.7),
      aridity=.65*math.sin(twopi*y/2.9e9+1.1)+.35*math.sin(twopi*y/17.3e9),
      productivity=max(.25,.78+.16*math.sin(twopi*y/.91e9+2.2)+.06*math.sin(twopi*y/7.7e9)),
      pathogen=.45+.30*(.5+.5*math.sin(twopi*y/.63e9+.4)),
      fragmentation=.45+.40*(.5+.5*math.sin(twopi*y/3.3e9+2.0)),
      resource_axis=math.sin(twopi*y/1.13e9)+.5*math.sin(twopi*y/5.9e9+.3),
      predator_pressure=0.0)

def resample(w,rng):
    n=len(w); p=(rng.random()+np.arange(n))/n; c=np.cumsum(w); c[-1]=1
    return np.searchsorted(c,p,'left')

def simulate(years=3e11,step=5e7,pop=6e9,n=4096,seed=20260929,checkpoint=5e9,out='results/fly-evolution-300b'):
    assert round(years/step)*step==years
    rng=np.random.default_rng(seed); steps=int(round(years/step))
    tr=rng.normal(0,.045,(n,10)); brain=rng.normal(0,.012,(n,4)); edge=rng.normal(0,.012,n); niche=rng.normal(0,.12,(n,3)); w=np.full(n,1/n)
    y=0.; rs=0; hist=[]
    def snap():
        e=env(y); assert e['predator_pressure']==0
        tm=w@tr; ts=np.sqrt(w@((tr-tm)**2)); bs=w@np.exp(brain); neurons=float(REG@bs); es=float(w@np.exp(edge)); edges=float(BASE_E*es*(neurons/BASE_N)**.88)
        return dict(year=y,population=float(pop),predator_pressure=0.0,environment=e,
          shannon=float(np.exp(-np.sum(w*np.log(w+1e-300)))),simpson=float(1/np.sum(w*w)),resamples=rs,
          traits={k:float(v) for k,v in zip(TN,tm)},trait_sd={k:float(v) for k,v in zip(TN,ts)},
          brain=dict(baseline_neurons=BASE_N,baseline_edges=BASE_E,mean_neurons=neurons,mean_edges=edges,region_scale={k:float(v) for k,v in zip(RN,bs)},edge_scale=es))
    hist.append(snap()); cp=max(1,int(round(checkpoint/step)))
    for i in range(1,steps+1):
        e=env(y); assert e['predator_pressure']==0
        th,dry,breadth,defense,disp,fec,dev,social,body,eff=tr.T; sensory,central,drives,motor=brain.T
        target=np.array([.85*e['resource_axis'],.75*e['fragmentation']-.35,.6*e['temperature']])
        fit=(-.24*(th-e['temperature'])**2-.20*(dry-e['aridity'])**2-.028*(th*th+dry*dry)
          -.18*np.sum((niche-target)**2,1)+.095*breadth*abs(e['resource_axis'])+.070*disp*e['fragmentation']
          +.080*central*(.25+abs(e['resource_axis']))+.035*sensory*e['fragmentation']
          +.11*defense*e['pathogen']-.030*defense**2+.10*fec*e['productivity']+.07*dev-.030*fec**2-.028*dev**2
          -.035*body**2+.020*body*(.2-e['aridity'])+.105*eff-.030*eff**2+.028*social*(1-e['fragmentation'])-.018*social**2
          -.090*((np.exp(brain)*(REG/BASE_N)).sum(1)-1)**2-.040*edge**2+.030*motor*e['fragmentation']+.018*drives*e['productivity'])
        c=fit-w@fit; bott=max(0,.72-e['productivity']); lw=np.log(w+1e-300)+2*c+rng.normal(0,.010+.060*bott,n); lw-=lw.max(); w=np.exp(lw); w/=w.sum()
        mf=float(w@fit); K=float(np.clip(6e9*(.45+1.35*e['productivity'])*math.exp(.10*mf),4e8,2.2e10)); g=.35*math.tanh(mf)+.28*math.log(K/max(pop,1)); pop=float(np.clip(pop*math.exp(np.clip(g,-.45,.45)),1,3e10))
        sev=1+1.8*bott
        if 1/np.sum(w*w)<.58*n:
            ix=resample(w,rng); tr=tr[ix].copy(); brain=brain[ix].copy(); edge=edge[ix].copy(); niche=niche[ix].copy(); w.fill(1/n); rs+=1
        tr+=rng.normal(0,.018*sev,tr.shape); niche+=rng.normal(0,.022*sev,niche.shape); brain+=rng.normal(0,.010*sev,brain.shape); edge+=rng.normal(0,.009*sev,n)
        jm=rng.random(brain.shape)<.0015*sev; brain+=jm*rng.normal(0,.12,brain.shape); je=rng.random(n)<.0012*sev; edge+=je*rng.normal(0,.11,n)
        np.clip(tr,-4,4,out=tr); np.clip(niche,-4.5,4.5,out=niche); np.clip(brain,math.log(.03),math.log(30),out=brain); np.clip(edge,math.log(.05),math.log(20),out=edge)
        y+=step
        if i%cp==0 or i==steps: hist.append(snap())
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    report=dict(model='accelerated_lineage_particle_macroevolution_v1',warning='Stochastic model, not a literal forecast. 300 Gyr exceeds natural Earth habitability; Earth-like habitability is maintained by assumption.',
      config=dict(years=years,step_years=step,steps=steps,initial_population=6e9,lineages=n,seed=seed,habitat_mode='maintained_habitable_earth'),
      baseline=dict(neurons=BASE_N,edges=BASE_E,region_neurons={k:int(v) for k,v in zip(RN,REG)}),algorithm=dict(explicit_individuals=False,predators=False,selection=True,mutation=True,brain_structure_mutation=True,drift=True,resource_competition=True,pathogens=True),initial=hist[0],final=hist[-1],history=hist)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'checkpoints.csv').open('w',newline='') as f:
        z=csv.writer(f); z.writerow(['year','population','predator_pressure','shannon','mean_neurons','mean_edges'])
        for s in hist:z.writerow([f"{s['year']:.0f}",f"{s['population']:.6g}",0,f"{s['shannon']:.6f}",f"{s['brain']['mean_neurons']:.3f}",f"{s['brain']['mean_edges']:.3f}"])
    assert report['final']['year']==years and all(x['predator_pressure']==0 for x in hist)
    print(json.dumps(report['final'],indent=2)); return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results/fly-evolution-300b'); p.add_argument('--lineages',type=int,default=4096); a=p.parse_args(); simulate(n=a.lineages,out=a.out)

#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import ImageFilter

import flybrain_ai_integration as base

SENSORY = ['VIS_R1R6','OLF_ORN_FOOD','MECH_BRISTLE','THERMO_WARM','GUS_GRN_SWEET','GUS_GRN_WATER']
TARGETS = ['MB_KC','LH_APP','CX_FC','GNG_DESC','MN_HEAD','MN_PROBOSCIS']
AVOID = ['GUS_GRN_BITTER','DRIVE_FATIGUE']


def sigmoid(x):
    return 1.0/(1.0+math.exp(-float(x)))


def aggregate_connectome(c):
    G=len(c.group_names)
    pg=c.group[c.pre].astype(np.int64); qg=c.group[c.post].astype(np.int64)
    idx=pg*G+qg
    signed=np.bincount(idx,weights=c.weight.astype(np.float64),minlength=G*G).reshape(G,G)
    absw=np.bincount(idx,weights=np.abs(c.weight).astype(np.float64),minlength=G*G).reshape(G,G)
    A=np.tanh(3.0*signed/(absw.sum(axis=1,keepdims=True)+1e-9))
    return A


def surrogate(A, stim, steps=9, leak=.72):
    x=np.zeros(A.shape[0],np.float64)
    for _ in range(steps):
        x=np.tanh(leak*x+stim+A.T@x)
    return x


def entropy(v):
    a=np.abs(v); s=a.sum()
    if s < 1e-12: return 0.0
    p=a/s; p=p[p>0]
    return float(-(p*np.log(p+1e-12)).sum()/math.log(max(2,len(v))))


def evolve(c,A,pop=40,generations=28,seed=20260929):
    rng=np.random.default_rng(seed)
    ids={n:c.group_names.index(n) for n in SENSORY if n in c.group_names and c.group_sizes[c.group_names.index(n)]>0}
    target=[c.group_names.index(n) for n in TARGETS if n in c.group_names and c.group_sizes[c.group_names.index(n)]>0]
    avoid=[c.group_names.index(n) for n in AVOID if n in c.group_names and c.group_sizes[c.group_names.index(n)]>0]
    dim=len(ids)+4
    mu=np.zeros(dim); mu[:len(ids)]=.7
    sigma=.65; archive=[]; history=[]; best=None
    keys=list(ids)
    for gen in range(generations):
        genomes=mu+rng.normal(0,sigma,size=(pop,dim))
        scores=[]; states=[]
        for g in genomes:
            stim=np.zeros(len(c.group_names))
            amps=1.8/(1+np.exp(-g[:len(keys)]))
            for n,a in zip(keys,amps): stim[ids[n]]=a
            x=surrogate(A,stim)
            central=float(np.mean(x[target])) if target else 0.0
            bad=float(np.mean(np.maximum(x[avoid],0))) if avoid else 0.0
            div=entropy(x)
            sat=float(np.mean(np.abs(x)>.97))
            novelty=0.0 if not archive else float(min(np.linalg.norm(x-z) for z in archive)/math.sqrt(len(x)))
            # Fitness rewards downstream activation, distributed activity and novelty, while penalizing saturation/avoid groups.
            score=2.4*central+.75*div+.35*novelty-.75*bad-.45*sat
            scores.append(score); states.append(x)
        order=np.argsort(scores)[::-1]; elite=max(5,pop//6)
        mu=genomes[order[:elite]].mean(0); sigma=max(.08,sigma*.91)
        bi=int(order[0]); archive.append(states[bi]); archive=archive[-20:]
        row={'generation':gen,'best':float(scores[bi]),'mean':float(np.mean(scores)),'sigma':float(sigma)}
        history.append(row)
        if best is None or scores[bi]>best['fitness']:
            best={'fitness':float(scores[bi]),'generation':gen,'genome':genomes[bi].tolist(),'state':states[bi].tolist()}
    best['sensory_names']=keys
    return best,history


def run_multi_lif(c,best,ticks=12):
    g=np.array(best['genome'],np.float64); names=best['sensory_names']
    amps=1.45/(1+np.exp(-g[:len(names)]))+.10
    stim=[]
    for n,a in zip(names,amps):
        gid=c.group_names.index(n); idx=np.flatnonzero(c.group==gid)
        # Limit giant visual population so one group cannot trivially dominate every generation.
        if idx.size>5000: idx=idx[::max(1,idx.size//5000)][:5000]
        stim.append((n,gid,idx,float(a)))
    v=np.zeros(c.n,np.float32); ref=np.zeros(c.n,np.uint8); hist=np.zeros((ticks,len(c.group_names)),np.int64)
    fired=np.zeros(c.n,bool)
    for t in range(ticks):
        v*=.95; r=ref>0; v[r]=0; ref[r]-=1
        if fired.any():
            ae=fired[c.pre]
            if ae.any():
                syn=np.bincount(c.post[ae],weights=c.weight[ae],minlength=c.n).astype(np.float32)
                v+=syn
        for _,_,idx,a in stim:
            live=ref[idx]==0; v[idx[live]]+=a
        fired=(ref==0)&(v>=1.0)
        fi=np.flatnonzero(fired)
        if fi.size:
            hist[t]=np.bincount(c.group[fi],minlength=len(c.group_names)); v[fi]=0; ref[fi]=3
        np.clip(v,-4,4,out=v)
    totals=hist.sum(0); topids=np.argsort(totals)[::-1][:12]
    top=[{'group':c.group_names[int(i)],'spikes':int(totals[int(i)])} for i in topids if totals[int(i)]>0]
    den=np.maximum(c.group_sizes,1)*ticks
    feat=np.log1p((totals.astype(np.float32)/den.astype(np.float32))*1000.0).astype(np.float32)
    return {'ticks':ticks,'total_fired':int(totals.sum()),'top_groups':top,'feature':feat,'hist':hist,
            'stimuli':[{'group':n,'amplitude':a,'neurons_stimulated':int(len(idx))} for n,_,idx,a in stim]}


class BrainPrefix(nn.Module):
    def __init__(self,brain_dim,hidden,prefix_len=8):
        super().__init__(); self.prefix_len=prefix_len; self.hidden=hidden
        self.net=nn.Sequential(nn.Linear(brain_dim,128),nn.Tanh(),nn.Linear(128,prefix_len*hidden))
        nn.init.normal_(self.net[-1].weight,std=.003); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x).view(x.shape[0],self.prefix_len,self.hidden)


def train_brain_prefix(brain,state,top,outdir,model_id='HuggingFaceTB/SmolLM2-135M-Instruct'):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    t0=time.time(); tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    adapter=BrainPrefix(len(brain),model.config.hidden_size,8)
    opt=torch.optim.AdamW(adapter.parameters(),lr=.025,weight_decay=1e-4)
    b=torch.tensor(brain,dtype=torch.float32).unsqueeze(0)
    toptext=', '.join(x['group'] for x in top[:6])
    target=(f"Photorealistic macro photograph of a fruit fly actively exploring ripe fruit, neural state dominated by {toptext}, "
            f"food-seeking drive {state['appetite']:.2f}, threat {state['threat']:.2f}, arousal {state['arousal']:.2f}, "
            "translucent wings, compound eyes, dew droplets, natural rim light, shallow depth of field, 100mm macro lens, microscopic biological detail.")
    prompt='Write one photorealistic Stable Diffusion prompt describing the current fruit-fly neural state. Output only the prompt.'
    pids=tok(prompt,return_tensors='pt').input_ids; tids=tok(target+(tok.eos_token or ''),return_tensors='pt').input_ids
    emb=model.get_input_embeddings(); losses=[]
    for step in range(18):
        opt.zero_grad(set_to_none=True); pref=adapter(b)
        pe=emb(pids).detach(); te=emb(tids).detach(); inp=torch.cat([pref,pe,te],1)
        labels=torch.full((1,inp.shape[1]),-100,dtype=torch.long); start=pref.shape[1]+pids.shape[1]; labels[:,start:start+tids.shape[1]]=tids
        o=model(inputs_embeds=inp,labels=labels,use_cache=False); o.loss.backward(); nn.utils.clip_grad_norm_(adapter.parameters(),1.0); opt.step(); losses.append(float(o.loss.detach()))
    # Autoregressive decode from the trained continuous prefix.
    genome_seed=int(abs(float(brain.sum()))*1e7)%2_000_000_000; torch.manual_seed(genome_seed)
    with torch.no_grad():
        pref=adapter(b); pe=emb(pids); o=model(inputs_embeds=torch.cat([pref,pe],1),use_cache=True); past=o.past_key_values; logits=o.logits[:,-1,:]; toks=[]
        for _ in range(90):
            nxt=torch.argmax(logits,dim=-1,keepdim=True); tid=int(nxt.item())
            if tid==tok.eos_token_id: break
            toks.append(tid); o=model(input_ids=nxt,past_key_values=past,use_cache=True); past=o.past_key_values; logits=o.logits[:,-1,:]
    generated=tok.decode(toks,skip_special_tokens=True).strip().replace('\n',' ')
    valid=len(generated.split())>=20 and 'fruit' in generated.lower()
    selected=generated if valid else target
    torch.save({'state_dict':adapter.state_dict(),'brain_dim':len(brain),'hidden':model.config.hidden_size,'prefix_len':8},outdir/'llm_brain_prefix.pt')
    info={'model':model_id,'base_model_parameters':sum(p.numel() for p in model.parameters()),'trained_adapter_parameters':sum(p.numel() for p in adapter.parameters()),
          'loss_start':losses[0],'loss_end':losses[-1],'loss_curve':losses,'generated':generated,'generated_valid':valid,'selected_prompt':selected,'train_target':target,'seconds':round(time.time()-t0,3)}
    (outdir/'llm_adapter_result.json').write_text(json.dumps(info,indent=2))
    del model,tok,adapter,opt
    return info


def image_proxy(img):
    gray=img.convert('L'); a=np.asarray(gray,dtype=np.float32)/255.0
    h=np.histogram(a,bins=64,range=(0,1))[0].astype(float); p=h/(h.sum()+1e-9); p=p[p>0]
    ent=float(-(p*np.log(p)).sum()/math.log(64)); edge=float(np.asarray(gray.filter(ImageFilter.FIND_EDGES),dtype=np.float32).mean()/255.0); contrast=float(a.std())
    return {'entropy':ent,'edge_density':edge,'contrast':contrast,'proxy_score':.45*ent+.30*edge+.25*contrast}


def evolve_images(prompt,best,outdir,model_id='segmind/tiny-sd'):
    from diffusers import DiffusionPipeline
    t0=time.time(); pipe=DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32); pipe=pipe.to('cpu')
    if hasattr(pipe,'enable_attention_slicing'): pipe.enable_attention_slicing()
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    g=np.array(best['genome']); base_guid=3.2+4.0*sigmoid(g[-2]); base_steps=6+int(4*sigmoid(g[-1]))
    neg='text, watermark, logo, cartoon, illustration, blurry, malformed insect, extra wings, low detail'
    candidates=[]
    for i,(dg,ds) in enumerate([(-.7,-1),(0,0),(.7,1)]):
        guidance=float(np.clip(base_guid+dg,2.5,8.0)); steps=int(np.clip(base_steps+ds,5,11)); seed=20260929+i*7919
        gen=torch.Generator(device='cpu').manual_seed(seed)
        im=pipe(prompt,negative_prompt=neg,num_inference_steps=steps,guidance_scale=guidance,height=256,width=256,generator=gen).images[0]
        path=outdir/f'sd_candidate_{i}.png'; im.save(path); score=image_proxy(im)
        candidates.append({'index':i,'guidance':guidance,'steps':steps,'seed':seed,'path':path.name,**score})
    win=max(candidates,key=lambda x:x['proxy_score']); src=outdir/win['path']; final=outdir/'sd_evolved_best.png'; final.write_bytes(src.read_bytes())
    info={'model':model_id,'prompt':prompt,'negative_prompt':neg,'candidates':candidates,'winner':win,'note':'Winner uses a low-cost visual complexity proxy, not a human aesthetic score.','seconds':round(time.time()-t0,3)}
    (outdir/'diffusion_evolution.json').write_text(json.dumps(info,indent=2)); return info


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--flybrain-dir',type=Path,default=Path('external/flybrain')); ap.add_argument('--out',type=Path,default=Path('artifacts/flybrain_evolution_v3')); args=ap.parse_args()
    random.seed(20260929); np.random.seed(20260929); torch.manual_seed(20260929); args.out.mkdir(parents=True,exist_ok=True)
    c=base.load_connectome(args.flybrain_dir/'data/connectome.bin.gz',args.flybrain_dir/'data/neuron_meta.json')
    A=aggregate_connectome(c); best,hist=evolve(c,A)
    with (args.out/'evolution_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['generation','best','mean','sigma']); w.writeheader(); w.writerows(hist)
    final=run_multi_lif(c,best,12)
    # Reuse the verified supervised neural readout so the connectome activity is mapped to interpretable drives.
    x,y,runs=base.dataset(c); readout,loss=base.train(x,y,args.out/'brain_readout.pt')
    with torch.no_grad(): statev=readout(torch.from_numpy(final['feature']).unsqueeze(0)).squeeze(0).numpy()
    state=base.words(statev)
    totals=final['hist'].sum(0)
    with (args.out/'brain_activity.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['group_id','group_name','spikes','neuron_count']); [w.writerow([i,n,int(totals[i]),int(c.group_sizes[i])]) for i,n in enumerate(c.group_names)]
    llm=train_brain_prefix(final['feature'],state,final['top_groups'],args.out)
    sd=evolve_images(llm['selected_prompt'],best,args.out)
    result={'connectome':{'neurons':c.n,'edges':c.e,'groups':len(c.group_names)},'evolution':{'generations':len(hist),'population':40,'best_fitness':best['fitness'],'best_generation':best['generation'],'best_genome':best['genome'],'sensory_names':best['sensory_names']},
            'full_lif':{'ticks':final['ticks'],'total_fired':final['total_fired'],'top_groups':final['top_groups'],'stimuli':final['stimuli'],'controller':state},
            'readout_training':{'samples':int(x.shape[0]),'loss_trace':loss},'llm_adapter':llm,'stable_diffusion_evolution':sd,
            'limitations':['The biological connectome supplies topology; the LIF dynamics are a computational approximation, not a complete living fly brain.','The LLM base weights are frozen; a brain-conditioned continuous-prefix adapter is trained.','Stable Diffusion base weights are frozen; evolutionary selection tunes prompting/sampling and selects among generated candidates.']}
    (args.out/'result_v3.json').write_text(json.dumps(result,indent=2))
    print('=== FLYBRAIN_EVOLUTION_V3 ==='); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()

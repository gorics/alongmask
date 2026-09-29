#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,json,platform,random,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

@dataclass
class Connectome:
    n:int; e:int; pre:np.ndarray; post:np.ndarray; weight:np.ndarray; region:np.ndarray; group:np.ndarray; group_names:list[str]; group_sizes:np.ndarray

def load_connectome(bin_gz:Path,meta_json:Path)->Connectome:
    t0=time.time()
    with gzip.open(bin_gz,'rb') as f: raw=f.read()
    h=np.frombuffer(raw,dtype='<u4',count=2,offset=0); n,e=int(h[0]),int(h[1])
    ed=np.dtype([('pre','<u4'),('post','<u4'),('weight','<f4')]); edges=np.frombuffer(raw,dtype=ed,count=e,offset=8)
    mo=8+e*12; md=np.dtype([('region','u1'),('group','<u2')]); nm=np.frombuffer(raw,dtype=md,count=n,offset=mo)
    pre=np.array(edges['pre'],dtype=np.uint32,copy=True); post=np.array(edges['post'],dtype=np.uint32,copy=True); w=np.array(edges['weight'],dtype=np.float32,copy=True)
    m=float(np.max(np.abs(w))) if e else 0.0
    if m>0: w=(w/m*0.15).astype(np.float32,copy=False)
    region=np.array(nm['region'],dtype=np.uint8,copy=True); group=np.array(nm['group'],dtype=np.uint16,copy=True)
    meta=json.loads(meta_json.read_text()); mg=max(g['id'] for g in meta['groups']); names=[f'GROUP_{i}' for i in range(mg+1)]; sizes=np.zeros(mg+1,dtype=np.int64)
    for g in meta['groups']: names[g['id']]=g['name']; sizes[g['id']]=int(g['neuron_count'])
    print(json.dumps({'event':'connectome_loaded','neurons':n,'edges':e,'groups':len(names),'seconds':round(time.time()-t0,3)}),flush=True)
    return Connectome(n,e,pre,post,w,region,group,names,sizes)

def run_lif(c:Connectome,stim:str,intensity:float,ticks:int=6)->dict:
    gid=c.group_names.index(stim); idx=np.flatnonzero(c.group==gid)
    if idx.size==0: raise ValueError(f'zero-size group: {stim}')
    v=np.zeros(c.n,np.float32); ref=np.zeros(c.n,np.uint8); hist=np.zeros((ticks,len(c.group_names)),np.int64); total=0
    for t in range(ticks):
        v*=0.95; ar=ref>0; ref[ar]-=1; v[idx]+=intensity
        fired=(v>=1.0)&(ref==0); fi=np.flatnonzero(fired)
        if fi.size:
            hist[t]=np.bincount(c.group[fi],minlength=len(c.group_names)); total+=int(fi.size); v[fi]=0.0; ref[fi]=3
            ae=fired[c.pre]
            if np.any(ae):
                syn=np.zeros(c.n,np.float32); np.add.at(syn,c.post[ae],c.weight[ae]); v+=syn
        np.clip(v,-4.0,4.0,out=v)
    den=np.maximum(c.group_sizes,1)*max(ticks,1); feat=hist.sum(0).astype(np.float32)/den.astype(np.float32); feat=np.log1p(feat*1000.0).astype(np.float32)
    gt=hist.sum(0); topids=np.argsort(gt)[::-1][:10]; top=[{'group':c.group_names[int(i)],'spikes':int(gt[int(i)])} for i in topids if gt[int(i)]>0]
    return {'stimulus':stim,'intensity':float(intensity),'ticks':ticks,'total_fired':total,'feature':feat,'spike_hist':hist,'top_groups':top}

class Readout(nn.Module):
    def __init__(self,d:int): super().__init__(); self.net=nn.Sequential(nn.Linear(d,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,4),nn.Tanh())
    def forward(self,x): return self.net(x)

def dataset(c):
    targets={'OLF_ORN_FOOD':[1,-.8,.25,0],'OLF_ORN_DANGER':[-.8,1,.9,0],'MECH_BRISTLE':[-.2,.75,1,0],'THERMO_WARM':[0,.1,.35,1],'THERMO_COOL':[0,.05,.25,-1],'VIS_R1R6':[0,-.2,.55,0]}
    xs=[]; ys=[]; runs=[]
    for name,y in targets.items():
        if name not in c.group_names or c.group_sizes[c.group_names.index(name)]==0: continue
        for q in (1.05,1.25):
            r=run_lif(c,name,q,5); xs.append(r['feature']); ys.append(np.array(y,np.float32)); runs.append({k:v for k,v in r.items() if k not in ('feature','spike_hist')})
    return np.stack(xs),np.stack(ys),runs

def train(xn,yn,path):
    torch.manual_seed(7); m=Readout(xn.shape[1]); x=torch.from_numpy(xn); y=torch.from_numpy(yn); opt=torch.optim.AdamW(m.parameters(),lr=2e-2,weight_decay=1e-4); trace=[]
    for ep in range(500):
        opt.zero_grad(set_to_none=True); p=m(x); loss=nn.functional.mse_loss(p,y); loss.backward(); opt.step()
        if ep%50==0 or ep==499: trace.append(float(loss.detach()))
    torch.save({'state_dict':m.state_dict(),'input_dim':xn.shape[1]},path); return m.eval(),trace

def words(s):
    a,t,r,h=[float(x) for x in s]
    return {'appetite':a,'threat':t,'arousal':r,'thermal':h,'mood':'seeking' if a>.25 else ('defensive' if t>.25 else 'exploratory'),'energy':'high' if r>.4 else 'calm','temperature_bias':'warm' if h>.25 else ('cool' if h<-.25 else 'neutral')}

def llm_prompt(state,top,model_id):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(model_id); model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32,low_cpu_mem_usage=True); model.eval()
    sys='Convert a fruit-fly connectome state into one vivid Stable Diffusion image prompt. Return only one sentence, no explanation.'
    usr=f"Brain state: {json.dumps(state)}. Active groups: {json.dumps(top[:6])}. Create a photorealistic macro-scene metaphor for this internal state."
    msgs=[{'role':'system','content':sys},{'role':'user','content':usr}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True) if getattr(tok,'chat_template',None) else sys+'\n'+usr+'\nPrompt:'
    inp=tok(text,return_tensors='pt')
    with torch.no_grad(): out=model.generate(**inp,max_new_tokens=72,do_sample=True,temperature=.75,top_p=.9,repetition_penalty=1.05,pad_token_id=tok.eos_token_id)
    new=out[0,inp['input_ids'].shape[1]:]; res=tok.decode(new,skip_special_tokens=True).strip().split('\n')[0].strip().strip('"')
    if len(res)<20: res=f"Photorealistic macro photograph of a fruit fly in a {state['mood']} state, {state['energy']} neural energy, {state['temperature_bias']} atmosphere, microscopic detail, dramatic natural light, shallow depth of field"
    return res,{'model':model_id,'output_chars':len(res)}

def stable_diffusion(prompt,model_id,out,steps):
    from diffusers import DiffusionPipeline
    pipe=DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32); pipe=pipe.to('cpu')
    if hasattr(pipe,'enable_attention_slicing'): pipe.enable_attention_slicing()
    g=torch.Generator(device='cpu').manual_seed(20260929); t0=time.time(); im=pipe(prompt,num_inference_steps=steps,guidance_scale=5.5,height=256,width=256,generator=g).images[0]; im.save(out)
    return {'model':model_id,'steps':steps,'width':im.width,'height':im.height,'seconds':round(time.time()-t0,3),'path':str(out)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--flybrain-dir',type=Path,default=Path('external/flybrain')); ap.add_argument('--out',type=Path,default=Path('artifacts')); ap.add_argument('--llm',default='HuggingFaceTB/SmolLM2-360M-Instruct'); ap.add_argument('--sd',default='segmind/tiny-sd'); ap.add_argument('--sd-steps',type=int,default=6); args=ap.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7); args.out.mkdir(parents=True,exist_ok=True)
    c=load_connectome(args.flybrain_dir/'data/connectome.bin.gz',args.flybrain_dir/'data/neuron_meta.json'); x,y,runs=dataset(c); model,losses=train(x,y,args.out/'brain_readout.pt')
    fr=run_lif(c,'OLF_ORN_FOOD',1.35,8)
    with torch.no_grad(): statev=model(torch.from_numpy(fr['feature']).unsqueeze(0)).squeeze(0).numpy()
    state=words(statev); totals=fr['spike_hist'].sum(0)
    with (args.out/'brain_activity.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['group_id','group_name','spikes','neuron_count']); [w.writerow([i,n,int(totals[i]),int(c.group_sizes[i])]) for i,n in enumerate(c.group_names)]
    prompt,li=llm_prompt(state,fr['top_groups'],args.llm); sd=stable_diffusion(prompt,args.sd,args.out/'flybrain_sd.png',args.sd_steps)
    result={'connectome':{'neurons':c.n,'edges':c.e,'groups':len(c.group_names)},'training':{'samples':int(x.shape[0]),'input_dim':int(x.shape[1]),'loss_trace':losses,'runs':runs},'final_brain_run':{'stimulus':fr['stimulus'],'intensity':fr['intensity'],'ticks':fr['ticks'],'total_fired':fr['total_fired'],'top_groups':fr['top_groups'],'controller':state},'llm':{**li,'prompt':prompt},'stable_diffusion':sd,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu'}}
    (args.out/'result.json').write_text(json.dumps(result,indent=2)); print('=== FINAL_RESULT ==='); print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()

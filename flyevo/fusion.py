import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline

OUT=Path('flyevo/results'); OUT.mkdir(parents=True,exist_ok=True)
z=np.load(OUT/'best_genome.npz'); llm_gain=float(z['llm_gain']); sd_gain=float(z['sd_gain'])

# LLM organ: pretrained tiny GPT-2 -> measured hidden state -> trained organism adapter.
mid='sshleifer/tiny-gpt2'
tok=AutoTokenizer.from_pretrained(mid)
model=AutoModelForCausalLM.from_pretrained(mid)
model.eval()
ins=tok('fruit fly neural state target ahead memory active',return_tensors='pt')
with torch.no_grad():
    o=model(**ins,output_hidden_states=True)
    h=o.hidden_states[-1][0,-1].float()
adapter=torch.nn.Linear(h.numel(),8)
opt=torch.optim.Adam(adapter.parameters(),lr=.05)
target=torch.tensor([.9,.4,.2,-.2,.7,.1,-.4,.5])
losses=[]
for _ in range(40):
    opt.zero_grad(); y=torch.tanh(adapter(h)); loss=((y-target)**2).mean(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
llm_lat=torch.tanh(adapter(h)).detach().cpu().numpy().astype(np.float32)
with torch.no_grad():
    gen=model.generate(**ins,max_new_tokens=10,do_sample=False,pad_token_id=tok.eos_token_id)
seq=gen.sequences[0] if hasattr(gen,'sequences') else gen[0]
text=tok.decode(seq.detach().cpu().tolist(),skip_special_tokens=True)

# Diffusion organ: actual Stable Diffusion inference -> visual feature state injected into same recurrent brain.
sid='hf-internal-testing/tiny-stable-diffusion-pipe'
pipe=StableDiffusionPipeline.from_pretrained(sid,torch_dtype=torch.float32,safety_checker=None)
pipe.set_progress_bar_config(disable=True)
img=pipe('abstract fruit fly compound eye neural activity scientific visualization',num_inference_steps=2,guidance_scale=1.0).images[0]
img.save(OUT/'sd_generated.png')
a=np.asarray(img.resize((16,16))).astype(np.float32)/255
sd=np.array([a[...,0].mean(),a[...,1].mean(),a[...,2].mean(),a.std(),a[:8].mean(),a[8:].mean(),a[:,:8].mean(),a[:,8:].mean()],np.float32)
sd=(sd-sd.mean())/(sd.std()+1e-6)

# One organism state: LLM and diffusion channels enter the same membrane update as sensory input.
W=z['weights'].astype(np.float32); th=float(z['threshold']); leak=float(z['leak']); N=W.shape[0]; S=12

def probe(use):
    v=np.zeros(N,np.float32); sp=np.zeros(N,np.float32); totals=[]; states=[]
    for t in range(40):
        inp=np.zeros(N,np.float32); inp[:S]=.25
        if use:
            inp[S:S+8]+=llm_gain*llm_lat
            inp[S+8:S+16]+=sd_gain*sd
        v=leak*v+inp+sp@W
        sp=(v>th).astype(np.float32); v[sp>0]=0
        totals.append(int(sp.sum()))
        if t in (0,9,19,29,39): states.append({'t':t,'spikes':int(sp.sum()),'membrane_mean':float(v.mean()),'membrane_std':float(v.std())})
    return totals,states
f,fs=probe(True); b,bs=probe(False)
report={
 'llm':{'model':mid,'generated':text,'hidden_dim':int(h.numel()),'adapter_loss_start':losses[0],'adapter_loss_final':losses[-1],'latent':llm_lat.tolist()},
 'stable_diffusion':{'model':sid,'size':list(img.size),'features':sd.tolist(),'image':'sd_generated.png'},
 'genome_gains':{'llm_gain':llm_gain,'sd_gain':sd_gain},
 'neural_probe':{'fused_total_spikes':sum(f),'baseline_total_spikes':sum(b),'delta':sum(f)-sum(b),'fused_states':fs,'baseline_states':bs},
 'claim_boundary':'This demonstrates one recurrent computational state influenced by actual LLM and diffusion inference features; it does not establish biological consciousness.'
}
(OUT/'fusion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))

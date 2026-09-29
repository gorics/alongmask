#!/usr/bin/env python3
"""V4 quality evolution: keep V3 connectome evolution + trained LLM prefix, reject repetitive text, use SD1.5 candidates."""
from __future__ import annotations
import json, math, time
from pathlib import Path
import torch

import flybrain_evolution_v3 as v3

_original_train = v3.train_brain_prefix


def repetition_score(text: str):
    toks=[x.strip('.,:;()').lower() for x in text.split() if x.strip()]
    if not toks: return 1.0,0.0
    unique=len(set(toks))/len(toks)
    bigrams=list(zip(toks,toks[1:])); mx=0
    if bigrams:
        from collections import Counter
        mx=max(Counter(bigrams).values())
    return unique,float(mx)


def quality_train_brain_prefix(brain,state,top,outdir,model_id='HuggingFaceTB/SmolLM2-135M-Instruct'):
    info=_original_train(brain,state,top,outdir,model_id)
    unique,max_bigram=repetition_score(info['generated'])
    text=info['generated'].lower()
    reject=(unique < .68 or max_bigram >= 3 or text.count('macro lens') >= 3 or len(info['generated']) < 100)
    if reject:
        info['selected_prompt']=info['train_target']
        info['generated_valid']=False
        info['quality_fallback_reason']={'unique_token_ratio':unique,'max_bigram_repeat':max_bigram,'macro_lens_mentions':text.count('macro lens')}
    else:
        info['quality_fallback_reason']=None
    (Path(outdir)/'llm_adapter_result.json').write_text(json.dumps(info,indent=2))
    return info


def sd15_evolve_images(prompt,best,outdir,model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'):
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    t0=time.time()
    pipe=DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32,use_safetensors=True)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe=pipe.to('cpu'); pipe.enable_attention_slicing()
    try: pipe.set_progress_bar_config(disable=True)
    except Exception: pass
    g=best['genome']; base_guid=5.8+1.5*v3.sigmoid(g[-2]); base_steps=9+int(2*v3.sigmoid(g[-1]))
    neg='cartoon, illustration, painting, text, watermark, logo, deformed insect, extra wings, extra legs, blurry, low detail, plastic toy'
    candidates=[]
    variants=[(-.55,-1,20260929),(0,0,20268848),(.55,1,20276767)]
    for i,(dg,ds,seed) in enumerate(variants):
        guidance=float(max(4.5,min(8.0,base_guid+dg))); steps=int(max(8,min(13,base_steps+ds)))
        gen=torch.Generator(device='cpu').manual_seed(seed)
        image=pipe(prompt,negative_prompt=neg,num_inference_steps=steps,guidance_scale=guidance,height=384,width=384,generator=gen).images[0]
        path=Path(outdir)/f'sd15_candidate_{i}.png'; image.save(path); score=v3.image_proxy(image)
        candidates.append({'index':i,'guidance':guidance,'steps':steps,'seed':seed,'path':path.name,**score})
    winner=max(candidates,key=lambda x:x['proxy_score'])
    src=Path(outdir)/winner['path']; final=Path(outdir)/'sd15_evolved_best.png'; final.write_bytes(src.read_bytes())
    info={'model':model_id,'prompt':prompt,'negative_prompt':neg,'candidates':candidates,'winner':winner,
          'note':'Three real SD1.5 generations compete using the same low-cost visual-complexity proxy; proxy score is not a human aesthetic judgment.',
          'seconds':round(time.time()-t0,3)}
    (Path(outdir)/'diffusion_evolution.json').write_text(json.dumps(info,indent=2))
    return info

v3.train_brain_prefix=quality_train_brain_prefix
v3.evolve_images=sd15_evolve_images

if __name__=='__main__':
    v3.main()

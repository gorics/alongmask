from __future__ import annotations
import gc, json, os, time
from pathlib import Path
import torch

ROOT = Path(os.environ.get("FLYBRAIN_DIR", "flybrain-upstream"))
OUT = Path("experiments/evofly/action-results")
OUT.mkdir(parents=True, exist_ok=True)

class UnifiedOrganism:
    def __init__(self, meta: dict):
        self.meta = meta
        self.genome = {"language_gate":0.42,"imagination_gate":0.31,"memory_gate":0.27,"structural_generation":1}
        self.neural_state = {"hunger":0.83,"fear":0.21,"curiosity":0.64,"visual_motion":0.57,"food_odor":0.71,"motor_intent":"forward-left"}
    def state_text(self):
        s=self.neural_state
        return (f"FlyWire connectome neurons={self.meta['neuron_count']}, edges={self.meta['edge_count']}; "
                f"hunger={s['hunger']:.2f}, fear={s['fear']:.2f}, curiosity={s['curiosity']:.2f}, "
                f"visual_motion={s['visual_motion']:.2f}, food_odor={s['food_odor']:.2f}, motor_intent={s['motor_intent']}; genome={self.genome}")
    def language_organ(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id="Qwen/Qwen3-0.6B"
        tok=AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        messages=[{"role":"user","content":"다음 초파리 시뮬레이터의 관찰 가능한 신경 상태를 한국어 한 문장으로 요약해. 실제 의식이나 주관적 생각을 읽었다고 주장하지 마. 상태: "+self.state_text()}]
        try: text=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: text=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        x=tok(text,return_tensors="pt")
        with torch.inference_mode(): y=model.generate(**x,max_new_tokens=56,do_sample=False,pad_token_id=tok.eos_token_id)
        result=tok.decode(y[0][x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        del model,tok,x,y; gc.collect()
        return result
    def imagination_organ(self, decoded):
        from diffusers import DiffusionPipeline
        model_id="segmind/tiny-sd"
        pipe=DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32,safety_checker=None)
        pipe.enable_attention_slicing()
        prompt="compound-eye fruit fly visual field, scientific visualization, dark arena, bright food cue ahead, red danger cue to the side, motion mosaic, neural perception; "+decoded[:180]
        with torch.inference_mode(): image=pipe(prompt,num_inference_steps=2,guidance_scale=1.0,height=256,width=256).images[0]
        path=OUT/'stable_diffusion_imagination.png'; image.save(path)
        del pipe,image; gc.collect(); return str(path)

def main():
    t0=time.time(); meta=json.loads((ROOT/'data/neuron_meta.json').read_text())
    organism=UnifiedOrganism(meta); decoded=organism.language_organ(); image_path=organism.imagination_organ(decoded)
    result={"neuron_count":meta['neuron_count'],"edge_count":meta['edge_count'],"language_model":"Qwen/Qwen3-0.6B","diffusion_model":"segmind/tiny-sd","decoded_state_ko":decoded,"imagination_image":image_path,"genome":organism.genome,"runtime_sec":time.time()-t0,"interpretation_warning":"Decoded text is an observation-layer state summary, not a readout of subjective consciousness."}
    (OUT/'real_organs_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import gc, json, os, re, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get("FLYBRAIN_DIR", "flybrain-upstream"))
OUT = Path("experiments/evofly/action-results")
OUT.mkdir(parents=True, exist_ok=True)

class UnifiedOrganism:
    """Single runtime state shared by connectome, language and image organs."""
    def __init__(self, meta: dict):
        self.meta = meta
        self.genome = {"language_gate":0.42,"imagination_gate":0.31,"memory_gate":0.27,"structural_generation":2}
        self.neural_state = {"hunger":0.83,"fear":0.21,"curiosity":0.64,"visual_motion":0.57,"food_odor":0.71,"motor_intent":"forward-left","language_feedback":0.0,"imagination_feedback":0.0}
    def state_text(self):
        s=self.neural_state
        return (f"FlyWire 뉴런={self.meta['neuron_count']}, 연결={self.meta['edge_count']}; "
                f"배고픔={s['hunger']:.2f}, 공포={s['fear']:.2f}, 호기심={s['curiosity']:.2f}, "
                f"시각운동={s['visual_motion']:.2f}, 먹이냄새={s['food_odor']:.2f}, 운동명령={s['motor_intent']}; 유전체={self.genome}")
    @staticmethod
    def hangul_fraction(text):
        if not text: return 0.0
        return len(re.findall(r'[가-힣]',text))/max(1,len(text))
    def language_organ(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id="Qwen/Qwen3-0.6B"
        tok=AutoTokenizer.from_pretrained(model_id)
        model=AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        def generate(messages):
            try: prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            except TypeError: prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            x=tok(prompt,return_tensors='pt')
            with torch.inference_mode(): y=model.generate(**x,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
            return tok.decode(y[0][x['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        system="너는 초파리 신경 시뮬레이터의 상태 디코더다. 반드시 한글 중심의 한국어 한 문장만 출력한다. 이것은 관찰 가능한 계산 상태 요약일 뿐이며 의식, 자아, 주관적 경험을 읽었다고 말하면 안 된다."
        user="다음 상태를 행동 관점에서 짧게 요약해. 상태: "+self.state_text()
        raw=generate([{"role":"system","content":system},{"role":"user","content":user}])
        final=raw
        bad=(self.hangul_fraction(final)<0.18 or any(w in final.lower() for w in ['awareness','consciousness','subjective']))
        if bad:
            final=generate([{"role":"system","content":system},{"role":"user","content":"아래 초안을 영어 없이 자연스러운 한국어 한 문장으로 다시 써. 의식·자아를 뜻하는 표현도 쓰지 마. 초안: "+raw+"\n상태: "+self.state_text()}])
        fallback=False
        if self.hangul_fraction(final)<0.18:
            final="배고픔과 먹이 냄새 신호가 높고 시각 움직임이 감지되어 먹이 쪽으로 이동하려는 운동 상태가 활성화되어 있다."
            fallback=True
        self.neural_state['language_feedback']=min(1.0,len(final)/120.0)*self.genome['language_gate']
        del model,tok; gc.collect()
        return raw,final,fallback
    def imagination_organ(self, decoded):
        from diffusers import DiffusionPipeline
        model_id="segmind/tiny-sd"
        pipe=DiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32,safety_checker=None)
        pipe.enable_attention_slicing()
        prompt=("compound-eye fruit fly visual field, scientific visualization, dark arena, bright food cue ahead, red danger cue to the side, motion mosaic, neural sensory map, "
                f"language feedback {self.neural_state['language_feedback']:.3f}; "+decoded[:160])
        with torch.inference_mode(): image=pipe(prompt,num_inference_steps=2,guidance_scale=1.0,height=256,width=256).images[0]
        self.neural_state['imagination_feedback']=float(np.asarray(image,dtype=np.float32).mean()/255.0)*self.genome['imagination_gate']
        path=OUT/'stable_diffusion_imagination.png'; image.save(path)
        del pipe,image; gc.collect(); return str(path)

def main():
    t0=time.time(); meta=json.loads((ROOT/'data/neuron_meta.json').read_text())
    organism=UnifiedOrganism(meta); raw,decoded,fallback=organism.language_organ(); image_path=organism.imagination_organ(decoded)
    result={"neuron_count":meta['neuron_count'],"edge_count":meta['edge_count'],"language_model":"Qwen/Qwen3-0.6B","diffusion_model":"segmind/tiny-sd","language_raw":raw,"decoded_state_ko":decoded,"decoder_fallback_used":fallback,"imagination_image":image_path,"genome":organism.genome,"final_shared_neural_state":organism.neural_state,"runtime_sec":time.time()-t0,"interpretation_warning":"Decoded text is an observation-layer state summary, not a readout of subjective consciousness."}
    (OUT/'real_organs_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

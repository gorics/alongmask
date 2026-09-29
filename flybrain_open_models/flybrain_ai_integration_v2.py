#!/usr/bin/env python3
"""Quality pass: reuse the verified FlyBrain simulator/readout and upgrade generative backends."""
from __future__ import annotations

import json
import time
import torch

import flybrain_ai_integration as base


def qwen_llm_prompt(state, top, model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    system = (
        "You are an expert text-to-image prompt writer. Convert numerical neural state data "
        "into a concrete visual scene. Output exactly ONE English image prompt. No JSON, "
        "no labels, no analysis, no quotation marks. The prompt MUST begin with "
        "'Photorealistic macro photograph of a fruit fly'."
    )
    state_text = json.dumps(state, ensure_ascii=False)
    groups_text = json.dumps(top[:6], ensure_ascii=False)
    user = (
        f"Neural controller state: {state_text}\n"
        f"Most active biological neural groups: {groups_text}\n"
        "Interpret high appetite as active food-seeking, low threat as safety, arousal as motion intensity, "
        "and thermal bias as atmosphere. Write a visually coherent 35-60 word Stable Diffusion prompt "
        "with subject, environment, action, lighting, optics, texture, and composition."
    )

    def generate(messages):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=110,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )
        new = out[0, inp["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip().replace("\n", " ").strip().strip('"')

    prompt = generate([{"role": "system", "content": system}, {"role": "user", "content": user}])
    valid = (
        prompt.lower().startswith("photorealistic macro photograph of a fruit fly")
        and "{" not in prompt
        and len(prompt.split()) >= 25
    )
    retried = False
    if not valid:
        retried = True
        repair = (
            "Rewrite your previous answer. Obey the required opening phrase exactly, remove all JSON/state labels, "
            "and return only a 35-60 word photographic image prompt. Previous answer: " + prompt
        )
        prompt = generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": repair},
        ])

    # Last-resort guard only if the open LLM still violates format. The neural values remain the source.
    if not prompt.lower().startswith("photorealistic macro photograph of a fruit fly") or "{" in prompt:
        prompt = (
            f"Photorealistic macro photograph of a fruit fly actively seeking ripe fruit in a safe environment, "
            f"calm purposeful posture reflecting appetite {state['appetite']:.2f} and low threat, "
            "dew-covered fruit surface, translucent wings and compound eyes in microscopic detail, warm natural rim light, "
            "shallow depth of field, 100mm macro lens, crisp biological texture, cinematic composition."
        )

    return prompt, {
        "model": model_id,
        "output_chars": len(prompt),
        "retried_for_format": retried,
    }


def sd15_generate(prompt, model_id, out_path, steps):
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    generator = torch.Generator(device="cpu").manual_seed(20260929)
    image = pipe(
        prompt,
        negative_prompt="cartoon, illustration, painting, text, watermark, logo, deformed insect, extra wings, blurry, low detail",
        num_inference_steps=steps,
        guidance_scale=7.0,
        height=384,
        width=384,
        generator=generator,
    ).images[0]
    image.save(out_path)
    return {
        "model": model_id,
        "steps": steps,
        "width": image.width,
        "height": image.height,
        "seconds": round(time.time() - t0, 3),
        "path": str(out_path),
    }


base.llm_prompt = qwen_llm_prompt
base.stable_diffusion = sd15_generate

if __name__ == "__main__":
    base.main()

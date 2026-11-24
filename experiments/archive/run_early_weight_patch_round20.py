"""Time-limited application of existing learned weight patches; no refit, no frequency runtime correction."""
from pathlib import Path
import torch,json,shutil,time,hashlib,numpy as np
from PIL import Image,ImageDraw
from diffusers import QwenImage21Pipeline
b=Path('/root/autodl-tmp/datasets');o=b/'early_output_weight_round20_20261118_codex';o.mkdir(exist_ok=False);shutil.copy2(__file__,o/'run_script.py');torch.set_num_threads(4)
patch=b/'output_head_distillation_round17_20261118_codex/output_head_patches.pt';data=torch.load(patch,map_location='cpu',weights_only=True)
tasks=[t for t in json.loads((b/'weight_edit_fresh_check_20261116_codex/tasks.json').read_text()) if t['suite']=='fresh'];conditions={'early_distill05':'distill05','early_distill10':'distill10','early_random05':'random_matched05'}
(o/'protocol.json').write_text(json.dumps(dict(tasks=tasks,conditions=list(conditions),patch_mapping=conditions,steps=40,cfg=1.,seed=42,resolution=1024,cache=False,weight_application_steps=list(range(12)),patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),hypothesis='Static patch causes late-timestep interference. Use exactly the same learned/random patches only during first12 steps, restore original after step11. No refitting or teacher correction at inference. Compare to matching all-step round17 conditions.',scope='Development ablation with time-dependent weight switching, not a single static model checkpoint or independent Astra reproduction.'),indent=2))
p=QwenImage21Pipeline.from_pretrained('/root/autodl-tmp/models/Qwen-Image-2.1',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');p.transformer.eval();w=p.transformer.proj_out.weight;original=data['original_weight'].to(w.device);assert torch.equal(w,original);records=[]
with torch.inference_mode():
 for t in tasks:
  for condition,patchname in ([('zero',None)] if t==tasks[0] else [])+list(conditions.items()):
   steps=[];started=time.monotonic()
   if patchname:w.copy_(data['variants'][patchname].to(w.device))
   def cb(pipe,i,timestep,kwargs):
    changed=not torch.equal(w,original);assert changed==(patchname is not None and i<12)
    steps.append(dict(step=i,edited_weights_used=changed))
    if i==11:w.copy_(original)
    return kwargs
   try:
    im=p(image=Image.open(t['source_image']).convert('RGB'),prompt=t['prompt'],num_inference_steps=40,true_cfg_scale=1.,output_resolution=1024,use_kv_cache=False,generator=torch.Generator('cuda').manual_seed(42),callback_on_step_end=cb,callback_on_step_end_tensor_inputs=['latents']).images[0]
   finally:w.copy_(original)
   assert torch.equal(w,original);im.save(o/f"{t['task_id']}_{condition}.png")
   if condition=='zero':
    same=np.array_equal(np.asarray(im),np.asarray(Image.open(t['control'])));assert same;(o/'zero_check.json').write_text(json.dumps(dict(pixel_identical=same)))
   records.append(dict(task_id=t['task_id'],condition=condition,seconds=time.monotonic()-started,weight_schedule=steps,original_restored=True));(o/'audit.json').write_text(json.dumps(records,indent=2));print('GENERATED',t['task_id'],condition,flush=True)
  sheet=Image.new('RGB',(2680,572),'#dddddd');d=ImageDraw.Draw(sheet)
  for j,(path,label) in enumerate([(t['source_image'],'SOURCE'),(t['control'],'native')]+[(o/f"{t['task_id']}_{c}.png",c) for c in conditions]):
   im=Image.open(path).convert('RGB');im.thumbnail((512,512));x=20+j*532;sheet.paste(im,(x,40));d.rectangle((x-1,39,x+512,552),outline='black');d.text((x,10),label,fill='black')
  sheet.save(o/f"comparison_{t['task_id']}.jpg",quality=97,subsampling=0)
(o/'completion.json').write_text(json.dumps(dict(completed=True,new_generations=13,method_outputs=12,diagnostic_outputs=1,original_weights_restored=True)))

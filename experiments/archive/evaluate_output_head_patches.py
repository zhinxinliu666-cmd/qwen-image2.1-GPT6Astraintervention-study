"""Evaluate static output-head weight patches using untouched native inference code."""
from pathlib import Path
import torch,json,time,hashlib,shutil,numpy as np
from PIL import Image,ImageDraw

def evaluate_patches(p,training_dir):
 b=Path('/root/autodl-tmp/datasets');o=b/'output_head_static_eval_round17_20261118_codex';o.mkdir(exist_ok=False);shutil.copy2(__file__,o/'run_script.py');patchfile=training_dir/'output_head_patches.pt';shutil.copy2(patchfile,o/patchfile.name)
 data=torch.load(patchfile,map_location='cpu',weights_only=True);w=p.transformer.proj_out.weight;original=data['original_weight'].to(w.device);assert torch.equal(w,original)
 tasks=[t for t in json.loads((b/'weight_edit_fresh_check_20261116_codex/tasks.json').read_text()) if t['suite']=='fresh'];conditions=list(data['variants']);assert len(conditions)==4
 (o/'protocol.json').write_text(json.dumps(dict(tasks=tasks,conditions=conditions,steps=40,cfg=1.,seed=42,resolution=1024,cache=False,weights_modified=['transformer.proj_out.weight'],inference='Native unmodified Qwen2.1 pipeline; no frequency hooks or CFG changes; original weights restored after every candidate.',training_split=json.loads((training_dir/'training_split.json').read_text()),patch_sha256=hashlib.sha256(patchfile.read_bytes()).hexdigest(),teacher='FreqEdit adaptation alpha1 compensated CFG1.2 (80 forwards); student native CFG1 (40 forwards); static ridge projection fit with fixed regularization and max5% delta norm; not human ground-truth training',selection='Predeclared strengths .25,.5,1 and BF16 norm-matched random direction control. Development images are disjoint from four training sources; no new holdout outputs used.'),indent=2))
 records=[]
 def run(t,condition):
  start=time.monotonic();im=p(image=Image.open(t['source_image']).convert('RGB'),prompt=t['prompt'],num_inference_steps=40,true_cfg_scale=1.,output_resolution=1024,use_kv_cache=False,generator=torch.Generator('cuda').manual_seed(42)).images[0];im.save(o/f"{t['task_id']}_{condition}.png");records.append(dict(task_id=t['task_id'],condition=condition,seconds=time.monotonic()-start,image_std=float(np.asarray(im).std()),actual_weight_delta_norm=float((w.float()-original.float()).norm())));(o/'audit.json').write_text(json.dumps(records,indent=2));print('STATIC_GENERATED',t['task_id'],condition,flush=True);return im
 with torch.inference_mode():
  im=run(tasks[0],'zero');same=np.array_equal(np.asarray(im),np.asarray(Image.open(tasks[0]['control'])));assert same;(o/'zero_check.json').write_text(json.dumps(dict(pixel_identical=same)))
  for t in tasks:
   for condition,edited in data['variants'].items():
    try:
     w.copy_(edited.to(w.device));assert not torch.equal(w,original);run(t,condition)
    finally:w.copy_(original)
    assert torch.equal(w,original)
   paths=[t['source_image'],t['control']]+[o/f"{t['task_id']}_{c}.png" for c in conditions];sheet=Image.new('RGB',(532*len(paths)+20,572),'#dddddd');draw=ImageDraw.Draw(sheet)
   for j,(path,label) in enumerate(zip(paths,['SOURCE','native']+conditions)):
    im=Image.open(path).convert('RGB');im.thumbnail((512,512));x=20+532*j;sheet.paste(im,(x,40));draw.rectangle((x-1,39,x+512,552),outline='black');draw.text((x,10),label,fill='black')
   sheet.save(o/f"comparison_{t['task_id']}.jpg",quality=97,subsampling=0)
  im=run(tasks[0],'restored');same=np.array_equal(np.asarray(im),np.asarray(Image.open(tasks[0]['control'])));assert same;(o/'restore_check.json').write_text(json.dumps(dict(original_weight_bitwise_restored=True,native_output_pixel_identical=True)))
 (o/'completion.json').write_text(json.dumps(dict(completed=True,new_generations=18,method_outputs=16,diagnostic_outputs=2,static_weights_modified=True,original_restored=True)))

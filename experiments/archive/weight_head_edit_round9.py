"""Gradient-free static output-projection edits selected by a read-only attention probe."""
from pathlib import Path
import json,shutil,time,hashlib
import torch,numpy as np
from PIL import Image,ImageDraw

def run_weight_head_edit(p,b):
 o=b/'weight_head_edit_20261118_codex';o.mkdir(exist_ok=False);shutil.copy2(__file__,o/'run_script.py')
 selected=json.loads((b/'attention_head_probe_20261118_codex/head_ranking.json').read_text())['selected']
 if not selected:
  (o/'completion.json').write_text(json.dumps(dict(completed=True,new_generations=0,reason='No heads passed frozen positive-across-sources criterion; no weight edit performed.')));return
 tasks=[t for t in json.loads((b/'weight_edit_fresh_check_20261116_codex/tasks.json').read_text()) if t['suite']=='fresh'];conditions=['aligned05','aligned10','random_matched10'];attns=[x.attn for x in p.transformer.transformer_blocks]
 chosen={}
 for h in selected:chosen.setdefault(h['layer'],[]).append(h['head'])
 rng=np.random.default_rng(20261118);random_heads={};snapshots={};indices={};head_dims={}
 for layer,heads in chosen.items():
  a=attns[layer];nh=a.heads;dim=a.to_out[0].weight.shape[1]//nh;head_dims[layer]=dim;pool=[h for h in range(nh) if h not in heads];assert len(pool)>=len(heads)
  random_heads[layer]=rng.choice(pool,size=len(heads),replace=False).tolist();ids=[i for h in sorted(heads+random_heads[layer]) for i in range(h*dim,(h+1)*dim)];idx=torch.tensor(ids,device=a.to_out[0].weight.device);indices[layer]=idx;snapshots[layer]=a.to_out[0].weight.detach().index_select(1,idx).clone()
 protocol=dict(tasks=tasks,conditions=conditions,chosen_heads=chosen,random_heads=random_heads,head_dimensions=head_dims,selection=selected,steps=40,seed=42,resolution=1024,cfg=1.,cache=False,weights_modified=True,mechanism='Scale static attention output-projection input columns belonging to selected heads. Affects all token types and all denoising steps; NOT a localized source-only attention intervention.',selected_scaling=[1.05,1.10],random_control='Same number of non-selected heads per selected layer. Scale chosen using only original weights to match selected1.10 Frobenius perturbation energy per layer, including BF16 quantization, within5%. No image-score tuning.',hypothesis='Heads with consistent spatially aligned reference attention may carry source geometry. Increasing their output contribution might preserve layout while keeping style; observational ranking alone is not causal evidence.',limitations='Four reused development images; selection uses step5 attention associations. Static edits also change text/reference representations and late steps. Negative or random-control-equivalent outcomes reject the proposed selective mechanism.')
 (o/'protocol.json').write_text(json.dumps(protocol,indent=2));records=[];weight_audits=[]
 def ids_for(layer,heads):return torch.tensor([i for h in heads for i in range(h*head_dims[layer],(h+1)*head_dims[layer])],device=attns[layer].to_out[0].weight.device)
 @torch.no_grad()
 def restore():
  for layer,idx in indices.items():attns[layer].to_out[0].weight.index_copy_(1,idx,snapshots[layer])
  assert all(torch.equal(attns[layer].to_out[0].weight.index_select(1,idx),snapshots[layer]) for layer,idx in indices.items())
 @torch.no_grad()
 def apply(cond):
  restore();patch={};audit=[]
  for layer in chosen:
   w=attns[layer].to_out[0].weight;si=ids_for(layer,chosen[layer]);sw=w.index_select(1,si).clone();goal=float(((sw.float()*1.10).to(w.dtype).float()-sw.float()).norm())
   heads=random_heads[layer] if cond=='random_matched10' else chosen[layer];ix=ids_for(layer,heads);old=w.index_select(1,ix).clone()
   if cond=='random_matched10':
    initial=goal/float(old.float().norm());candidates=[]
    for factor in np.linspace(.98,1.02,41):
     gain=1+initial*float(factor);new=(old.float()*gain).to(w.dtype);delta=float((new.float()-old.float()).norm());candidates.append((abs(delta-goal),gain,delta))
    _,gain,delta=min(candidates);assert abs(delta-goal)/max(goal,1e-12)<.05
   else:gain=1.05 if cond=='aligned05' else 1.10
   new=(old.float()*gain).to(w.dtype);w.index_copy_(1,ix,new);assert torch.isfinite(new).all();delta=new.float()-old.float()
   audit.append(dict(layer=layer,heads=heads,gain=gain,changed_scalars=int((new!=old).sum()),delta_frobenius=float(delta.norm()),relative_delta_frobenius=float(delta.norm()/old.float().norm()),matched_strong_selected_delta=goal))
   patch[str(layer)]=dict(column_indices=ix.cpu(),original=old.cpu(),edited=new.cpu())
  return patch,audit
 def generate(t,c):
  start=time.monotonic()
  with torch.inference_mode():im=p(image=Image.open(t['source_image']).convert('RGB'),prompt=t['prompt'],num_inference_steps=40,true_cfg_scale=1.,output_resolution=1024,use_kv_cache=False,generator=torch.Generator('cuda').manual_seed(42)).images[0]
  im.save(o/f"{t['task_id']}_{c}.png");records.append(dict(task_id=t['task_id'],condition=c,seconds=time.monotonic()-start));(o/'audit.json').write_text(json.dumps(records,indent=2));print('WEIGHT GENERATED',t['task_id'],c,flush=True);return im
 try:
  restore();z=generate(tasks[0],'zero');same=np.array_equal(np.asarray(z),np.asarray(Image.open(tasks[0]['control'])));assert same;(o/'zero_check.json').write_text(json.dumps(dict(pixel_identical=same)))
  for t in tasks:
   for c in conditions:
    patch,audit=apply(c)
    if t==tasks[0]:
     torch.save(patch,o/f'{c}_weight_patch.pt');weight_audits.append(dict(condition=c,layers=audit));(o/'weight_audit.json').write_text(json.dumps(weight_audits,indent=2))
    generate(t,c);restore()
   paths=[Path(t['source_image']),Path(t['control'])]+[o/f"{t['task_id']}_{c}.png" for c in conditions];sheet=Image.new('RGB',(512*len(paths),552),'white');d=ImageDraw.Draw(sheet)
   for j,(path,label) in enumerate(zip(paths,['SOURCE','baseline']+conditions)):
    im=Image.open(path).convert('RGB');im.thumbnail((512,512));sheet.paste(im,(j*512,40));d.text((j*512+8,8),label,fill='black')
   sheet.save(o/f"comparison_{t['task_id']}.jpg",quality=95)
 finally:
  restore();(o/'restoration.json').write_text(json.dumps(dict(all_modified_columns_bitwise_restored=True,original_checkpoint_files_not_written=True)))
 (o/'completion.json').write_text(json.dumps(dict(completed=True,new_generations=13,static_weights_modified=True,original_checkpoint_files_not_written=True)))
 print('WEIGHT HEAD EDIT COMPLETE',flush=True)

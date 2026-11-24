from pathlib import Path
import inspect,importlib,textwrap,types,json,shutil,time,sys,hashlib
import torch,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/datasets/external_baselines/freqedit_libs')
from FreqEdit_utils_official import correct_with_wavelet_guidance,adaptive_weight,dwt_pyramid,build_image_from_dwt_pyramid
from diffusers import QwenImage21Pipeline
torch.set_num_threads(4);b=Path('/root/autodl-tmp/datasets');o=b/'output_head_distillation_round17_20261118_codex';o.mkdir(exist_ok=False);shutil.copy2(__file__,o/'run_script.py');shutil.copy2(b/'FreqEdit_utils_official.py',o/'wavelet_utils_official.py')
# Numerical check on CPU before loading the model.
x=torch.randn(1,4,64,64,generator=torch.Generator().manual_seed(42));y=build_image_from_dwt_pyramid(dwt_pyramid(x,2,'db4'),'db4');err=float((x-y).abs().max());assert err<1e-5
(o/'wavelet_check.json').write_text(json.dumps(dict(max_abs_reconstruction_error=err,wavelet='db4',levels=2)))
from distill_output_head_helpers import setup_distillation,solve_distillation
shutil.copy2(b/'distill_output_head_helpers.py',o/'distill_output_head_helpers.py')
shutil.copy2(b/'evaluate_output_head_patches.py',o/'evaluate_output_head_patches.py')
train_sources=json.loads((b/'art_holdout_round7_20261118/sources.json').read_text())[:4]
train_ids={x['source_id'] for x in train_sources}
tasks=[t for t in json.loads((b/'art_holdout_round7_20261118/baseline_seed42/protocol.json').read_text())['tasks'] if t['source_id'] in train_ids];assert len(tasks)==12
fresh=[t for t in json.loads((b/'weight_edit_fresh_check_20261116_codex/tasks.json').read_text()) if t['suite']=='fresh']
pixel_hash=lambda path:hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()
train_hashes={pixel_hash(t['source_image']) for t in tasks};assert len(train_hashes)==4
assert train_hashes.isdisjoint({pixel_hash(t['source_image']) for t in fresh})
assert train_hashes.isdisjoint({t['pixel_sha256'] for t in json.loads((b/'art_photo_extension_round16_20261118/sources.json').read_text())})
assert train_hashes.isdisjoint({pixel_hash(t['source_image']) for t in json.loads((b/'art_photo_holdout_round10_20261118/baseline_seed42/protocol.json').read_text())['tasks']})
(o/'training_split.json').write_text(json.dumps(dict(selection='First four sources in pre-existing round7 frozen manifest; all three styles',source_ids=sorted(train_ids),pixel_hashes=sorted(train_hashes),disjoint_from_development_round10_and_round16=True),indent=2))
conditions={'freq10_guided12':(1.,True,1.2)}
(o/'protocol.json').write_text(json.dumps(dict(tasks=tasks,conditions=list(conditions),base_alpha_compensation=conditions,gamma=1.6,wavelet='db4',levels=2,injection_steps=12,compensation_period=4,skip_compensation_first_steps=2,steps=40,cfg=1.2,model_forwards_per_teacher=80,seed=42,resolution=1024,cache=False,reference='https://github.com/FreqEdit/FreqEdit',adaptation='Official wavelet/adaptive-weight functions, actual Qwen2.1 scheduler sigmas and unpatched spatial latent layout. Paper is older Qwen multi-turn editing; this is Qwen2.1 single-turn artistic adaptation, not original benchmark reproduction.',utility_sha256=hashlib.sha256((b/'FreqEdit_utils_official.py').read_bytes()).hexdigest(),weights_modified=False),indent=2))
p=QwenImage21Pipeline.from_pretrained('/root/autodl-tmp/models/Qwen-Image-2.1',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda');p.transformer.eval();distill_hook=setup_distillation(p,o);mod=importlib.import_module(QwenImage21Pipeline.__module__);s=textwrap.dedent(inspect.getsource(QwenImage21Pipeline.__call__))
s=s.replace('    # 5. Denoising loop',"""    assert do_true_cfg==(self.freq_cfg['cfg']>1)
    assert input_images_latents.shape == latents.shape
    correction_buffer=torch.zeros_like(latents,dtype=torch.float32)
    accumulated_count=0
    self.freq_audit=[]
    # 5. Denoising loop""")
needle='            latents_dtype = latents.dtype';assert s.count(needle)==1
inject="""            baseline_velocity=self.output_capture[0]['output'][:,-noise_pred.shape[1]:].float()
            self.teacher_delta=noise_pred.float()-baseline_velocity
            if self.freq_cfg['alpha'] > 0 and i<12:
                sigma=self.scheduler.sigmas[i].float().to(device)
                ds=(self.scheduler.sigmas[i+1]-self.scheduler.sigmas[i]).float().to(device)
                reference_velocity=(latents.float()-input_images_latents.float())/sigma
                vp=self._unpack_latents(noise_pred.float(),height,width,self.vae_scale_factor).squeeze(2)
                vr=self._unpack_latents(reference_velocity,height,width,self.vae_scale_factor).squeeze(2)
                weight=adaptive_weight(vr,vp,self.freq_cfg['alpha'],gamma=1.6)
                corrected4=correct_with_wavelet_guidance(vp,vr,weight,wavelet='db4',levels=2)
                corrected=self._pack_latents(corrected4,corrected4.shape[0],corrected4.shape[1],corrected4.shape[2],corrected4.shape[3])
                assert torch.isfinite(corrected).all()
                if self.freq_cfg['compensate'] and i>1:
                    correction_buffer+=ds*(noise_pred.float()-corrected)
                    accumulated_count+=1
                self.freq_audit.append(dict(step=i,sigma=float(sigma),alpha_mean=float(weight.mean()),alpha_max=float(weight.max()),relative_velocity_change=float((corrected-noise_pred.float()).norm()/noise_pred.float().norm())))
                noise_pred=corrected.to(noise_pred.dtype)
                self.teacher_delta=noise_pred.float()-baseline_velocity
"""
s=s.replace(needle,inject+needle)
needle2='            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]';assert s.count(needle2)==1
s=s.replace(needle2,needle2+"""
            if self.freq_cfg['compensate'] and (accumulated_count==4 or i==11):
                latents=(latents.float()+correction_buffer).to(latents_dtype)
                self.teacher_delta+=correction_buffer/ds
                correction_buffer.zero_();accumulated_count=0
            self.distill_accumulate(i,self.teacher_delta)
""")
ns=dict(mod.__dict__);ns.update(correct_with_wavelet_guidance=correct_with_wavelet_guidance,adaptive_weight=adaptive_weight);exec(compile(s,'<freqedit-qwen21>','exec'),ns);call=types.MethodType(ns['__call__'],p);(o/'pipeline_modified.py').write_text(s);records=[]
with torch.inference_mode():
 for task_index,t in enumerate(tasks):
  p.distill_task_index=task_index
  for c,(alpha,comp,cfg) in ([('zero',(0.,False,1.))] if t==tasks[0] else [])+list(conditions.items()):
   p.collect_distill=(c!='zero');p.freq_cfg=dict(alpha=alpha,compensate=comp,cfg=cfg);assert len(p.output_capture)==0;start=time.monotonic();im=call(image=Image.open(t['source_image']).convert('RGB'),prompt=t['prompt'],num_inference_steps=40,true_cfg_scale=cfg,negative_prompt="" if cfg>1 else None,output_resolution=1024,use_kv_cache=False,generator=torch.Generator('cuda').manual_seed(42)).images[0];im.save(o/f"{t['task_id']}_{c}.png")
   if c=='zero':
    same=np.array_equal(np.asarray(im),np.asarray(Image.open(t['control'])));assert same;(o/'zero_check.json').write_text(json.dumps(dict(pixel_identical=same)))
   records.append(dict(task_id=t['task_id'],condition=c,seconds=time.monotonic()-start,steps=p.freq_audit));(o/'audit.json').write_text(json.dumps(records,indent=2));print('GENERATED',t['task_id'],c,flush=True)
  paths=[Path(t['source_image']),Path(t['control'])]+[o/f"{t['task_id']}_{c}.png" for c in conditions];sheet=Image.new('RGB',(512*len(paths),552),'white');d=ImageDraw.Draw(sheet)
  for j,(path,label) in enumerate(zip(paths,['SOURCE','baseline']+list(conditions))):
   im=Image.open(path).convert('RGB');im.thumbnail((512,512));sheet.paste(im,(j*512,40));d.text((j*512+8,8),label,fill='black')
  sheet.save(o/f"comparison_{t['task_id']}.jpg",quality=95)
distill_hook.remove();solve_distillation(p,o)
(o/'completion.json').write_text(json.dumps(dict(completed=True,new_generations=13,teacher_outputs=12,static_weight_patches_fitted=True,patches_applied_during_collection=False)))

from evaluate_output_head_patches import evaluate_patches
evaluate_patches(p,o)

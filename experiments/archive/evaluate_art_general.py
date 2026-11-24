"""Multi-style exploratory evaluation. CPU avoids competing with generation GPU."""
from pathlib import Path
import sys,json,csv,statistics,shutil
import torch,numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel,AutoImageProcessor
sys.path.insert(0,'/root/autodl-tmp/datasets/external_baselines/clipstyler/libs')
import clip
torch.set_num_threads(4)
pth=json.loads(Path('/root/autodl-tmp/models/evaluation/dinov2-small/evaluation_provenance.json').read_text())['local_path']
dino=AutoModel.from_pretrained(pth,local_files_only=True).eval();proc=AutoImageProcessor.from_pretrained(pth,local_files_only=True)
cm,cp=clip.load('ViT-L/14',device='cpu',jit=False);cm.eval()
mean=torch.tensor(proc.image_mean).view(1,3,1,1);std=torch.tensor(proc.image_std).view(1,3,1,1)
style_names={'watercolor':'a watercolor painting','oil':'an oil painting','sketch':'a pencil sketch'}
style_keys=list(style_names)+['photo'];style_phrases=list(style_names.values())+['a photograph']
@torch.inference_mode()
def txt(strings):return F.normalize(cm.encode_text(clip.tokenize(strings,truncate=True)).float(),dim=-1)
@torch.inference_mode()
def enc(path):
 im=Image.open(path).convert('RGB');a=np.asarray(im.resize((560,560),Image.Resampling.BICUBIC),dtype=np.float32).copy()/255
 d=F.normalize(dino(pixel_values=(torch.from_numpy(a).permute(2,0,1)[None]-mean)/std).last_hidden_state[0].float(),dim=-1)
 c=F.normalize(cm.encode_image(cp(im)[None]).float(),dim=-1)
 return d,c
with torch.inference_mode():
 styles=txt(style_phrases);cache={}
 for arg in sys.argv[1:]:
  out=Path(arg);proto=json.loads((out/'protocol.json').read_text());assert json.loads((out/'completion.json').read_text())['completed'];rows=[]
  for t in proto['tasks']:
   sp=t['source_image']
   if sp not in cache:cache[sp]=enc(sp)
   sd,sc=cache[sp];ss=sd[1:]@sd[1:].T
   style=t.get('style','watercolor');idx=style_keys.index(style)
   caption=t['original_prompt'].replace('[','').replace(']','');text=txt([caption,style_names[style]+' of '+caption]);td=text[1:]-text[:1]
   for cond,path in [('copy',sp),('control',t['control'])]+[(c,str(out/f"{t['task_id']}_{c}.png")) for c in proto['conditions']]:
    if path not in cache:cache[path]=enc(path)
    d,c=cache[path];s=(c@styles.T)[0];delta=c-sc
    rows.append(dict(task_id=t['task_id'],source_id=t.get('source_id',t['task_id']),style=style,condition=cond,dino_patch=float((sd[1:]*d[1:]).sum(-1).mean()),dino_cls=float((sd[0]*d[0]).sum()),dino_relation_mse=float(((d[1:]@d[1:].T)-ss).square().mean()),clip_l14_source=float((sc*c).sum()),clip_l14_style_margin=float(s[idx]-s[3]),clip_l14_style_vs_other=float(s[idx]-torch.cat([s[:idx],s[idx+1:]]).max()),clip_l14_target=float((c*text[1:]).sum()),clip_l14_direction=float(F.cosine_similarity(delta,td).item()) if float(delta.norm())>1e-10 else None))
   print(out.name,t['task_id'],flush=True)
  meta=dict(dino_model=pth,dino_preprocessing='RGB bicubic560x560 normalized per official processor',relation_formula='Mean squared difference of cosine self-similarity matrices of normalized non-CLS DINOv2-small patch tokens. Exploratory structural proxy, not original PnP metric implementation.',clip_model='OpenAI CLIP ViT-L/14',style_phrases=style_phrases,device='cpu',limitations='CLIP proxy not human judgment; scores on custom artistic instructions are not official PIE scores; copy is diagnostic only',rows=rows)
  (out/'general_metrics.json').write_text(json.dumps(meta,indent=2));shutil.copy2(__file__,out/'evaluate_art_general.py')
  with (out/'general_metrics.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

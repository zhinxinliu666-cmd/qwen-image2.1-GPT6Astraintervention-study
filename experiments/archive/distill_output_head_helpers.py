"""Sufficient-statistics ridge fit for a static output-projection delta; no ground-truth claim."""
import torch,json,hashlib

def setup_distillation(p,o):
 w=p.transformer.proj_out.weight
 assert tuple(w.shape)==(64,4096),tuple(w.shape)
 p.original_output_weight=w.detach().clone();p.distill_stats={};p.distill_sample_audit=[];p.collect_distill=False;p.distill_task_index=0;p.output_capture=[]
 for group in ['train','validation_early','validation_late']:
  p.distill_stats[group]={'xx':torch.zeros(4096,4096,device='cuda',dtype=torch.float32),'xy':torch.zeros(4096,64,device='cuda',dtype=torch.float32),'yy':torch.zeros((),device='cuda',dtype=torch.float32),'n':0}
 def hook(module,args,output):p.output_capture.append({'features':args[0].detach(),'output':output.detach()})
 handle=p.transformer.proj_out.register_forward_hook(hook)
 def accumulate(step,target):
  assert len(p.output_capture)==(2 if p.freq_cfg.get('cfg',1)>1 else 1),len(p.output_capture)
  features=p.output_capture[0]['features'];p.output_capture=[]
  if not p.collect_distill:return
  y=target[0].float();x=features[0,-y.shape[0]:].float();assert x.shape==(y.shape[0],4096) and y.shape[1]==64
  seed=20261118+p.distill_task_index*40+step;ix=torch.randperm(x.shape[0],generator=torch.Generator().manual_seed(seed))[:320].to(x.device)
  for group,ids in [('train',ix[:256]),('validation_early' if step<12 else 'validation_late',ix[256:])]:
   xx=x[ids];yy=y[ids];a=p.distill_stats[group];a['xx'].addmm_(xx.T,xx);a['xy'].addmm_(xx.T,yy);a['yy']+=(yy*yy).sum();a['n']+=len(ids)
  p.distill_sample_audit.append({'task_index':p.distill_task_index,'step':step,'sample_seed':seed,'target_rms':float(y.square().mean().sqrt()),'feature_rms':float(x.square().mean().sqrt())})
 p.distill_accumulate=accumulate
 return handle

def solve_distillation(p,o):
 st=p.distill_stats['train'];n=st['n'];assert n==12*40*256,n
 c=st['xx'].double()/n;b=st['xy'].double()/n;lam=.01*float(c.diag().mean());assert lam>0
 c.diagonal().add_(lam);delta=torch.linalg.solve(c,b).T.float();assert torch.isfinite(delta).all()
 w0=p.original_output_weight;raw_norm=float(delta.norm());max_norm=.05*float(w0.float().norm());scale=min(1.,max_norm/max(raw_norm,1e-12));delta*=scale
 def fit_metrics(dw):
  result={}
  for group,s in p.distill_stats.items():
   pred=(dw@s['xx']*dw).sum();cross=(dw.T*s['xy']).sum();base=float(s['yy']);sse=float(pred-2*cross+s['yy']);result[group]={'tokens':s['n'],'zero_delta_sse':base,'fitted_sse':sse,'mse_per_scalar':sse/(s['n']*64),'fraction_of_zero_delta_sse':sse/base if base>1e-12 else None}
  return result
 variants={};metrics={}
 for strength in [.25,.5,1.]:
  name=f'distill{str(strength).replace(".","")}'
  edited=(w0.float()+strength*delta).to(w0.dtype);actual=edited.float()-w0.float();variants[name]=edited.cpu();metrics[name]={'strength':strength,'actual_weight_delta_norm':float(actual.norm()),'relative_weight_delta_norm':float(actual.norm()/w0.float().norm()),'changed_scalars':int((edited!=w0).sum()),'fit':fit_metrics(actual)}
 # Equal-norm BF16 random-direction control matched to strength0.5 after quantization.
 target=metrics['distill05']['actual_weight_delta_norm'];assert target>1e-8
 target_rows=(variants['distill05'].to(w0.device).float()-w0.float()).norm(dim=1)
 rnd=torch.randn(w0.shape,generator=torch.Generator('cuda').manual_seed(20261118),device='cuda',dtype=torch.float32);rnd/=rnd.norm(dim=1,keepdim=True)
 lo=torch.zeros_like(target_rows);hi=2*target_rows
 for _ in range(30):
  mid=(lo+hi)/2;actual=(w0.float()+mid[:,None]*rnd).to(w0.dtype).float()-w0.float();below=actual.norm(dim=1)<target_rows
  lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
 edited=(w0.float()+((lo+hi)/2)[:,None]*rnd).to(w0.dtype);actual=edited.float()-w0.float();rel=abs(float(actual.norm())-target)/target;rowrel=float(((actual.norm(dim=1)-target_rows).abs()/target_rows.clamp_min(1e-12)).max());assert rel<.005 and rowrel<.005,(rel,rowrel)
 variants['random_matched05']=edited.cpu();metrics['random_matched05']={'actual_weight_delta_norm':float(actual.norm()),'relative_weight_delta_norm':float(actual.norm()/w0.float().norm()),'changed_scalars':int((edited!=w0).sum()),'relative_norm_match_error':rel,'maximum_per_row_relative_norm_match_error':rowrel,'fit':fit_metrics(actual)}
 assert torch.equal(p.transformer.proj_out.weight,p.original_output_weight),'Collection must leave model weights untouched'
 torch.save({'original_weight':w0.cpu(),'ridge_delta_float32':delta.cpu(),'variants':variants},o/'output_head_patches.pt')
 torch.save({group:{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in a.items()} for group,a in p.distill_stats.items()},o/'sufficient_statistics.pt')
 (o/'fit_summary.json').write_text(json.dumps({'training_tokens':n,'ridge_lambda':lam,'ridge_relative_trace_factor':.01,'raw_delta_norm':raw_norm,'max_relative_weight_delta':.05,'norm_clip_scale':scale,'variants':metrics,'limitations':'Teacher-generated local velocity targets, not human ground-truth artistic images. Token-validation uses disjoint tokens from the same training sources; it is not unseen-source validation. Static rollout must be tested separately.'},indent=2))
 (o/'sample_audit.json').write_text(json.dumps(p.distill_sample_audit,indent=2))

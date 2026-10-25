"""Frozen J source/input/protocol, missing-key GPU collection, then SQLite controller runs."""
from pathlib import Path
from datetime import datetime
import argparse,collections,hashlib,itertools,json,os,platform,shutil,sys,traceback
os.environ['USE_TF']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
R=Path(__file__).resolve().parents[1];D=R/'raw_results/materialized_j_v1';A=R/'reports/current_revision'
def now():return datetime.now().astimezone().isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def rec(p):return {'path':p.relative_to(R).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def validate():
 for x in load(D/'input_manifest.json')+load(D/'source_manifest.json'):assert sha(R/x['path'])==x['sha256'],x['path']
 for n,h in load(D/'prepared.json')['files'].items():assert sha(D/n)==h,n
def prepare():
 assert not D.exists();D.mkdir(parents=True);A.mkdir(parents=True,exist_ok=True)
 H=R/'raw_results/live_h_v1';streams=load(H/'streams.json');contexts=load(H/'contexts.json');pool={}
 for line in (H/'model_calls.jsonl').read_text(encoding='utf-8').splitlines():
  x=json.loads(line);key=x['text_sha256']+'|'+x['context'];assert key not in pool or pool[key]==x['value'];pool[key]=x['value']
 requests={}
 for st in streams:
  for e in st['events']:
   for row in e['rows']:
    if e['rule']=='positive_and_top' and row['isTopCritic']!='True':continue
    k=row['text_sha256']+'|'+contexts[e['context_alias']]
    if k not in pool:requests[k]={'key':k,'text_sha256':row['text_sha256'],'context_alias':e['context_alias'],'text':row['reviewText']}
 cfg={'time':now(),'methods':['CF','SC','CR','WCR','WQCR'],'streams':3,'queries':['filter','join'],'fee_caps':[16,80],'write_caps':{'filter':[32,96,512],'join':[1200,2400,8192]},'audit_fee':1.,'refresh_fee':.05,'fixed_before_J_predictions_and_policy_execution':True,'prediction_pool':'316 unique previously executed H keys plus92 missing required keys collected before J controllers;controllers see returns only after payment','missing_predictions_expected':92,'total_required_keys':408,'new_semantic_texts':0,'post_hoc_workload_design':True,'seed':2026101903,'facts':'supplied H labels;not a human oracle','write_unit':'One inserted/deleted membership in explicit candidate or published result table. Includes intake,paid candidate deltas and final publication. Excludes SQL index/cache/metadata writes,bytes,IO,CPU and transactions themselves.','event_intake':'If exact candidate intake delta cannot fit,metadata/version advance but candidate/published tables remain untouched and event refuses.','protected':'Fee reserve if initially affordable. WCR/WQCR also preserve joint reserve only when both reserves fit after intake. Otherwise common immediate worst-write/fee guards apply;no completion guarantee.','publication':'Exact margins+current version+both budgets;refusal retains old physical published table but current-epoch read returns unavailable.','no_formal_latency':True,'no_external_engine_reproduction':True,'formal_submission':False}
 assert len(requests)==92;save(D/'protocol.json',cfg);save(D/'missing_prediction_requests.json',[requests[k] for k in sorted(requests)]);save(D/'prior_prediction_pool.json',pool)
 files=[H/n for n in ['streams.json','contexts.json','labels_EVALUATION_ONLY.json','model_calls.jsonl','protocol.json']]+sorted((R/'models/qwen2.5-0.5b-instruct').glob('*'))
 save(D/'input_manifest.json',[rec(p) for p in files if p.is_file()]);(D/'source_snapshot').mkdir();src=[]
 for n in ['d03_materialized_j.py','run_d03_materialized_j.py','verify_d03_materialized_j.py']:
  p=R/'src'/n;q=D/'source_snapshot'/n;shutil.copyfile(p,q);src.append({**rec(q),'origin':'src/'+n})
 save(D/'source_manifest.json',src);save(D/'environment_prepare.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'remote':False})
 save(D/'prepared.json',{'time':now(),'files':{n:sha(D/n) for n in ['protocol.json','missing_prediction_requests.json','prior_prediction_pool.json','input_manifest.json','source_manifest.json']},'launch_commands':['Anaconda Python src/run_d03_materialized_j.py collect','bundled Python src/run_d03_materialized_j.py run','bundled Python src/verify_d03_materialized_j.py'],'log_policy':'stdout/stderr for each phase kept in this run directory;failures retained;no overwrite'})
 print('J prepared:92 missing keys;180 controller trajectories expected;not run')
def collect():
 validate();assert not (D/'prediction_pool.json').exists();assert not (D/'new_GPU_calls.jsonl').exists()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2026101903);torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();assert torch.cuda.mem_get_info()[0]>2.5*1024**3
 cfg=load(R/'raw_results/live_h_v1/protocol.json');md=R/cfg['model'];tok=AutoTokenizer.from_pretrained(md,local_files_only=True);tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();ids=[tok.encode(x,add_special_tokens=False) for x in ['Yes','No']];assert all(len(x)==1 for x in ids)
 pool=load(D/'prior_prediction_pool.json');rows=[]
 with (D/'new_GPU_calls.jsonl').open('w',encoding='utf-8') as out:
  for r in load(D/'missing_prediction_requests.json'):
   p=cfg['prompts'][r['context_alias']];rendered=tok.apply_chat_template([{'role':'system','content':p['system']},{'role':'user','content':p['user'].format(text=r['text'])}],tokenize=False,add_generation_prompt=True);enc=tok([rendered],return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
   with torch.inference_mode():v=model(**enc).logits[0,-1,[ids[0][0],ids[1][0]]].float()
   assert torch.isfinite(v).all();b=int(v[0]>=v[1]);pool[r['key']]=b;x={k:r[k] for k in ['key','text_sha256','context_alias']};x.update(value=b,actual_GPU_forward=True,Yes_No_logits=v.cpu().tolist(),input_ids_sha256=hashlib.sha256(json.dumps(enc['input_ids'][0].cpu().tolist(),separators=(',',':')).encode()).hexdigest());out.write(json.dumps(x,separators=(',',':'))+'\n');out.flush();rows.append(x)
 assert len(rows)==92;save(D/'prediction_pool.json',pool);save(D/'prediction_collection.json',{'time':now(),'status':'GPU_COLLECTION_FINISHED','new_GPU_calls':len(rows),'prior_unique_keys':len(load(D/'prior_prediction_pool.json')),'pool_keys':len(pool),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'GPU':torch.cuda.get_device_name(),'dtype':'bfloat16','batch_size':1,'max_input_tokens':512,'shared_host':True,'predictions_generated_before_policy_runs':True});print('Actual new GPU calls',len(rows),'pool',len(pool))
def run():
 validate();assert (D/'prediction_collection.json').is_file() and not (D/'execution_receipt.json').exists()
 from d03_materialized_j import State,SQL,choose,digest
 cfg=load(D/'protocol.json');H=R/'raw_results/live_h_v1';streams=load(H/'streams.json');ctxs=load(H/'contexts.json');gold=load(H/'labels_EVALUATION_ONLY.json');pool=load(D/'prediction_pool.json');(D/'trajectories').mkdir();(D/'databases').mkdir();allrows=[];total=collections.Counter()
 for st,query,bf,bw,method in itertools.product(streams,cfg['queries'],cfg['fee_caps'],[0,1,2],cfg['methods']):
  bw=cfg['write_caps'][query][bw];tid=f"{st['id']}_{query}_f{bf}_w{bw}_{method}";facts={};pred={};last={};db=SQL(D/'databases'/(tid+'.sqlite'),query);events=[]
  for e in st['events']:
   s=State(query,e['rows'],ctxs[e['context_alias']],e['rule'],facts,pred,last,db.rows('published'));pubbefore=digest(db.rows('published'));fee=0.;accepted,write,intake_w=db.intake(s,e['epoch'],bw);fee_protected=bf>=s.U;protected=method in {'WCR','WQCR'} and fee_protected and accepted and write+s.delta+2*s.V<=bw;steps=[];tests=0;representatives=0;reason='intake_write_cap' if not accepted else None
   initial={'cert':s.cert,'U':s.U,'V':s.V,'delta':s.delta,'intake_required':intake_w,'intake_applied':accepted,'intake_writes':write,'joint_protected':protected,'fee_protected':fee_protected}
   if accepted:
    while not s.qualified() or (method in {'WCR','WQCR'} and s.delta>bw-write):
     a,n,rep=choose(s,method,bf-fee,bw-write,fee_protected,protected,cfg['refresh_fee'],cfg['audit_fee']);tests+=n;representatives+=rep
     if a is None:reason='no_admissible_action';break
     i,kind=a;k=s.e[i][0];b=gold[k] if kind=='audit' else pool[k+'|'+s.ctx];branch=s.branch(a,b);before=s.cert;fee+=cfg['audit_fee'] if kind=='audit' else cfg['refresh_fee']
     if kind=='audit':assert k not in facts;facts[k]=b
     else:assert (k,s.ctx) not in pred;pred[k,s.ctx]=b;last[k]=b
     s.apply(a,b);actual=db.paid(s,branch['w']);write+=actual;assert s.cert==tuple(branch['cert']) and s.delta==branch['delta'];assert fee<=bf+1e-8 and write<=bw
     steps.append({'rid':i,'key':k,'kind':kind,'bit':b,'fee_total':fee,'candidate_writes':actual,'write_total':write,'cert_before':before,'cert_after':s.cert,'delta':s.delta,'U':s.U,'V':s.V,'answer_sha256':digest(s.answer)})
     if protected:assert s.qualified() or (fee+s.U<=bf+1e-8 and write+s.delta+2*s.V<=bw)
     assert len(steps)<=2*len(e['rows'])
   published=False;pub_w=0
   if accepted and s.qualified():published,pub_w=db.publish(s,e['epoch'],bw-write);write+=pub_w;reason=None if published else 'publication_write_cap'
   answer=db.read(e['epoch']);assert (answer is not None)==published;assert db.read(e['epoch']+1000) is None
   if not published:assert digest(db.rows('published'))==pubbefore
   if protected:assert published
   ev={'epoch':e['epoch'],'kind':e['event_kind'],'nonempty':bool(e['rows']),'fee_cap':bf,'write_cap':bw,'initial':initial,'steps':steps,'fee':fee,'writes':write,'publication_writes':pub_w,'published':published,'refusal_reason':reason,'final_cert':s.cert,'answer_sha256':digest(s.answer),'published_table_sha256':digest(db.rows('published')),'branch_tests':tests,'representative_visits':representatives,'wrong_epoch_rejected':True}
   events.append(ev);total['events']+=1;total['steps']+=len(steps);total['writes']+=write;total['published']+=published;total['protected_events']+=protected
  assert db.c.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.c.close();t={'id':tid,'stream':st['id'],'query':query,'method':method,'fee_cap':bf,'write_cap':bw,'events':events,'database':f'raw_results/materialized_j_v1/databases/{tid}.sqlite','database_sha256':sha(D/'databases'/(tid+'.sqlite'))};save(D/'trajectories'/(tid+'.json'),t);allrows.append({'id':tid,'stream':st['id'],'query':query,'method':method,'fee_cap':bf,'write_cap':bw,'nonempty':sum(x['nonempty'] for x in events),'served':sum(x['nonempty'] and x['published'] for x in events),'fee':sum(x['fee'] for x in events),'writes':sum(x['writes'] for x in events),'steps':sum(len(x['steps']) for x in events),'branch_tests':sum(x['branch_tests'] for x in events)});print(tid,allrows[-1]['served'],flush=True)
 save(D/'summary.json',allrows);save(D/'execution_receipt.json',{'time':now(),'status':'EXECUTION_FINISHED_INDEPENDENT_VERIFICATION_PENDING','trajectories':len(allrows),'counts':dict(total),'new_GPU_during_controllers':0,'new_GPU_prediction_collection':92,'prediction_pool_sha256':sha(D/'prediction_pool.json'),'post_hoc_workload':True,'formal_submission':False});print('J controller trajectories',len(allrows),dict(total))
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','collect','run']);args=ap.parse_args()
try:{'prepare':prepare,'collect':collect,'run':run}[args.action]()
except BaseException as e:
 if D.exists():save(D/(args.action+'_failure.json'),{'time':now(),'error':str(e),'traceback':traceback.format_exc(),'partial_preserved':True})
 raise

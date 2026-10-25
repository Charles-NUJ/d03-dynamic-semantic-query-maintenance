"""Frozen R experiment: controlled variables and D03-specialized literature-inspired baselines."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse, collections, hashlib, itertools, json, os, platform, shutil, sys, time, traceback

os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
R=Path(__file__).resolve().parents[1]; D=R/'raw_results/controlled_r_v2'; A=R/'reports/controlled_r_current'
METHODS=['CF','SC','CR','WCR','WQCR','FRANK','SQP','LRANK','LQP','PGAIN','COST']
LEARNED={'LRANK','LQP','PGAIN'}
JOINT_SAFE={'WCR','WQCR','FRANK','SQP','LRANK','LQP','PGAIN','COST'}

def now(): return datetime.now().astimezone().isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rec(p):
    p=Path(p); return {'path':p.relative_to(R).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}

def validate():
    p=load(D/'prepared.json')
    assert sha(D/'protocol.json')==p['protocol_sha256']
    assert sha(D/'input_manifest.json')==p['input_manifest_sha256']
    assert sha(D/'source_manifest.json')==p['source_manifest_sha256']
    for x in load(D/'input_manifest.json')+load(D/'source_manifest.json'):
        assert sha(R/x['path'])==x['sha256'],x['path']

def prepare():
    assert not D.exists(), 'controlled_r_v2 already exists'
    D.mkdir(parents=True); A.mkdir(exist_ok=True); (D/'source_snapshot').mkdir()
    cfg={
      'time':now(),'status':'FROZEN_BEFORE_R_CONTROLLED_EXECUTION','methods':METHODS,
      'streams':3,'queries':['filter','join'],'fee_caps':[16,80],
      'write_caps':{'filter':[32,96,512],'join':[1200,2400,8192]},
      'audit_fee':1.0,'refresh_fee':0.05,'seed':2026102206,'deterministic':True,
      'trajectories':396,'events':3564,'same_initial_conditions':True,'same_return_tables':True,
      'factorial':{
        'probability':['fixed_0.5','paid_ridge'],'planner':['linear_rank','simplex_QP'],
        'cells':{'FRANK':['fixed_0.5','linear_rank'],'SQP':['fixed_0.5','simplex_QP'],
                 'LRANK':['paid_ridge','linear_rank'],'LQP':['paid_ridge','simplex_QP']},
        'same_state_counterfactual':'Every LQP decision records all four cells before the paid update.'},
      'structural_baselines':{
        'CF':'audit-to-quality then cheap refresh','SC':'refresh first','CR':'two-return certified release',
        'WCR':'write-aware raw actions','WQCR':'write-aware exact quotient'},
      'specialized_baselines':{
        'PGAIN':'Our D03 proxy/acquisition specialization: paid-ridge flip estimates, one-step physical objective, exact gates. Inspired by proxy-guided sampling and adaptive acquisition; not UQE, SUPG, BARGAIN, ActiveClean, or OLLA reproduction.',
        'COST':'Our D03 cost-based specialization: fee, worst publication commit, then stable order under exact gates. Inspired by semantic cost-based planning; not LOTUS, Abacus, JEVDB, or another engine reproduction.'},
      'controlled_variables':['same streams and row updates','same supplied fact and prediction return tables','same fee/write caps','same initial empty caches','same deterministic tie order','same quality and lineage tests','same hard publication fee/write limits'],
      'joint_safe_methods':sorted(JOINT_SAFE),
      'weak_baseline_boundary':'CF, SC, and CR retain the original quality-first control and may refuse at the final write gate even when the initial sufficient joint condition holds.',
      'noncontrolled_after_divergence':'End-to-end policies generate endogenous cache and learner histories. Pure factor effects use the same-state LQP counterfactual records.',
      'statistics':'Exact paired descriptive differences over 36 stream/query/budget cells; no iid p-values because budgets share three streams.',
      'no_new_texts':True,'new_GPU_calls':0,'new_annotations':0,'external_code_executed':False,
      'formal_submission':False}
    save(D/'protocol.json',cfg)
    files=[R/'raw_results/live_h_v1'/n for n in ['streams.json','contexts.json','labels_EVALUATION_ONLY.json']]+[R/'raw_results/materialized_j_v1/prediction_pool.json',R/'raw_results/materialized_j_v1/protocol.json']
    save(D/'input_manifest.json',[rec(p) for p in files]); snaps=[]
    for n in ['d03_controlled_r.py','run_d03_controlled_r.py','verify_d03_controlled_r.py','d03_adaptive_k.py','d03_materialized_j.py']:
        src=R/'src'/n; dst=D/'source_snapshot'/n; shutil.copyfile(src,dst)
        snaps.append({**rec(dst),'origin':'src/'+n,'origin_sha256':sha(src)})
    save(D/'source_manifest.json',snaps)
    import numpy as np
    save(D/'environment.json',{'time':now(),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'BLAS_threads':1,'remote':False,'GPU_calls':0})
    save(D/'prepared.json',{'time':now(),'protocol_sha256':sha(D/'protocol.json'),'input_manifest_sha256':sha(D/'input_manifest.json'),'source_manifest_sha256':sha(D/'source_manifest.json'),'frozen_before_execution':True,'launch':['python src/run_d03_controlled_r.py run','python src/verify_d03_controlled_r.py']})
    print('PREPARED_R_CONTROLLED',len(METHODS),'methods',flush=True)

def run():
    validate(); assert not (D/'execution_receipt.json').exists() and not (D/'trajectories').exists()
    from d03_controlled_r import State,SQL,digest,Learner,choose
    cfg=load(D/'protocol.json'); H=R/'raw_results/live_h_v1'; streams=load(H/'streams.json'); ctxs=load(H/'contexts.json'); gold=load(H/'labels_EVALUATION_ONLY.json'); pool=load(R/'raw_results/materialized_j_v1/prediction_pool.json')
    (D/'trajectories').mkdir(); (D/'databases').mkdir(); summary=[]; counts=collections.Counter(); start=time.perf_counter()
    for st,q,bf,ix,method in itertools.product(streams,cfg['queries'],cfg['fee_caps'],range(3),cfg['methods']):
        bw=cfg['write_caps'][q][ix]; tid=f"{st['id']}_{q}_f{bf}_w{bw}_{method}"; facts={}; pred={}; last={}; learner=Learner(); db=SQL(D/'databases'/(tid+'.sqlite'),q); events=[]; elapsed=0.0
        for event in st['events']:
            s=State(q,event['rows'],ctxs[event['context_alias']],event['rule'],facts,pred,last,db.rows('published')); pubbefore=digest(db.rows('published')); fee=0.; accepted,w,intake=db.intake(s,event['epoch'],bw); fp=bf>=s.U; jp=method in JOINT_SAFE and accepted and fp and w+s.delta+2*s.V<=bw; steps=[]; reason=None if accepted else 'intake_write_cap'; tests=0
            initial={'cert':s.cert,'U':s.U,'V':s.V,'delta':s.delta,'intake_required':intake,'intake_applied':accepted,'intake_writes':w,'joint_protected':jp,'fee_protected':fp,'rows':len(event['rows'])}
            if accepted:
                while not s.qualified() or (method in JOINT_SAFE and s.delta>bw-w):
                    t0=time.perf_counter(); action,plan=choose(s,method,bf-fee,bw-w,fp,jp,bf,bw,learner); elapsed+=time.perf_counter()-t0; tests+=plan['branch_tests']
                    if action is None: reason='no_admissible_action'; break
                    i,kind=action; key=s.e[i][0]; bit=gold[key] if kind=='audit' else pool[key+'|'+s.ctx]; branch=s.branch(action,bit); before=s.cert
                    feedback=learner.update(s,action,bit) if method in LEARNED else None
                    fee += 1.0 if kind=='audit' else 0.05
                    if kind=='audit': facts[key]=bit
                    else: pred[key,s.ctx]=bit; last[key]=bit
                    s.apply(action,bit); dw=db.paid(s,branch['w']); w+=dw
                    assert s.cert==tuple(branch['cert']) and s.delta==branch['delta'] and fee<=bf+1e-8 and w<=bw
                    if jp: assert s.qualified() or (fee+s.U<=bf+1e-8 and w+s.delta+2*s.V<=bw)
                    steps.append({'rid':i,'key':key,'kind':kind,'bit':bit,'fee_total':fee,'candidate_writes':dw,'write_total':w,'cert_before':before,'cert_after':s.cert,'delta':s.delta,'U':s.U,'V':s.V,'answer_sha256':digest(s.answer),'plan':plan,'paid_feedback':feedback})
                    counts['paid_actions']+=1; counts['factorial_states']+=int('controlled_cells' in plan)
                    if plan.get('mode')=='QP': counts['QP_solves']+=1; counts['QP_iterations']+=plan['iterations']; counts['QP_nonconverged']+=not plan['converged']
                    assert len(steps)<=2*len(event['rows'])
            published=False; pw=0
            if accepted and s.qualified(): published,pw=db.publish(s,event['epoch'],bw-w); w+=pw; reason=None if published else 'publication_write_cap'
            assert (db.read(event['epoch']) is not None)==published and db.read(event['epoch']+1000) is None
            if not published: assert digest(db.rows('published'))==pubbefore
            if jp: assert published
            events.append({'epoch':event['epoch'],'kind':event['event_kind'],'nonempty':bool(event['rows']),'fee_cap':bf,'write_cap':bw,'initial':initial,'steps':steps,'fee':fee,'writes':w,'publication_writes':pw,'published':published,'refusal_reason':reason,'final_cert':s.cert,'answer_sha256':digest(s.answer),'published_table_sha256':digest(db.rows('published')),'branch_tests':tests,'wrong_epoch_rejected':True})
            counts['events']+=1; counts['published']+=published; counts['refused']+=not published; counts['protected_events']+=jp
        assert db.c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'; db.c.close()
        dbp=D/'databases'/(tid+'.sqlite'); tr={'id':tid,'stream':st['id'],'query':q,'method':method,'fee_cap':bf,'write_cap':bw,'events':events,'database':dbp.relative_to(R).as_posix(),'database_sha256':sha(dbp),'controller_elapsed_debug_seconds':elapsed}; save(D/'trajectories'/(tid+'.json'),tr)
        summary.append({'id':tid,'stream':st['id'],'query':q,'method':method,'fee_cap':bf,'write_cap':bw,'nonempty':sum(x['nonempty'] for x in events),'served':sum(x['nonempty'] and x['published'] for x in events),'fee':sum(x['fee'] for x in events),'writes':sum(x['writes'] for x in events),'steps':sum(len(x['steps']) for x in events),'branch_tests':sum(x['branch_tests'] for x in events),'controller_elapsed_debug_seconds':elapsed})
        print(tid,summary[-1]['served'],round(elapsed,3),flush=True)
    save(D/'summary.json',summary); save(D/'execution_receipt.json',{'time':now(),'status':'R_CONTROLLED_EXECUTION_FINISHED_SEPARATE_VERIFICATION_PENDING','trajectories':len(summary),'counts':dict(counts),'wall_seconds':time.perf_counter()-start,'new_GPU_calls':0,'formal_submission':False,'no_latency_claim':True})
    print(json.dumps(dict(counts)),flush=True)

ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','run']); args=ap.parse_args()
try: {'prepare':prepare,'run':run}[args.action]()
except BaseException as e:
    if D.exists(): save(D/(args.action+'_failure.json'),{'time':now(),'error':repr(e),'traceback':traceback.format_exc(),'partial_preserved':True})
    raise

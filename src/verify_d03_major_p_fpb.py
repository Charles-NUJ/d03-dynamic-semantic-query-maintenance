"""Independent record and database verification for revision P FPB experiment."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sqlite3

R=Path(__file__).resolve().parents[1]; D=R/'raw_results/major_p_fpb_v1'; OUT=R/'derived/major_p_fpb_v1'; REP=R/'reports/major_p_current'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x): p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

p=load(D/'prepared.json')
assert sha(D/'protocol.json')==p['protocol_sha256'] and sha(D/'streams.json')==p['streams_sha256'] and sha(D/'input_manifest.json')==p['input_manifest_sha256'] and sha(D/'source_manifest.json')==p['source_manifest_sha256']
for x in load(D/'input_manifest.json')+load(D/'source_manifest.json'): assert sha(R/x['path'])==x['sha256'],x['path']
cfg=load(D/'protocol.json'); gold=load(R/'inputs/financial_phrasebank_review_v1/reference_labels_EVALUATION_ONLY.json'); contexts=load(D/'contexts.json'); streams={x['id']:x for x in load(D/'streams.json')}
pred={}; accuracy={}
for alias,ctx in contexts.items():
    table=load(D/f'{alias}_predictions.json'); assert set(table)==set(gold)
    assert all(v['label'] in [0,1] and len(v['probabilities'])==2 and abs(sum(v['probabilities'])-1)<1e-6 for v in table.values())
    pred[ctx]={k:v['label'] for k,v in table.items()}; accuracy[alias]=sum(table[k]['label']==gold[k] for k in gold)/len(gold)
summary=load(D/'summary.json'); assert len(summary)==36
trajs={x['id']:x for x in (load(p) for p in sorted((D/'trajectories').glob('*.json')))}; assert len(trajs)==36 and len(list((D/'databases').glob('*.sqlite')))==36
unsafe=0; budget_violations=0; db_errors=[]; event_count=0; actions=0
for tr in trajs.values():
    dbp=R/tr['database']; assert sha(dbp)==tr['database_sha256']
    con=sqlite3.connect(f'file:{dbp.as_posix()}?mode=ro',uri=True); q=con.execute('PRAGMA quick_check').fetchone()[0]; con.close()
    if q!='ok': db_errors.append({'path':str(dbp),'quick_check':q})
    stream=streams[tr['stream']]
    for event,spec in zip(tr['events'],stream['events']):
        event_count+=1; actions+=len(event['steps']); budget_violations+=int(event['fee']>tr['fee_cap']+1e-8 or event['writes']>tr['write_cap'])
        if event['published']:
            rows={r['reviewId']:r for r in spec['rows']}; P=set(event['published_rows']); Q={rid for rid,r in rows.items() if gold[r['text_sha256']]==1}; tp=len(P&Q); precision=tp/len(P) if P else 1.; recall=tp/len(Q) if Q else 1.
            unsafe+=int(precision<.9-1e-12 or recall<.9-1e-12)
        if event['initial']['joint_protected']: assert event['published']

pairs=[]
for stream,fee,write in ((s,f,w) for s in streams for f in cfg['fee_caps'] for w in cfg['write_caps']):
    a=trajs[f'{stream}_f{fee}_w{write}_WCR']; b=trajs[f'{stream}_f{fee}_w{write}_WQCR']
    trace=lambda t:[[(x['rid'],x['kind'],x['bit']) for x in e['steps']] for e in t['events']]
    assert trace(a)==trace(b)
    assert [(e['published'],e['fee'],e['writes'],e['published_rows']) for e in a['events']]==[(e['published'],e['fee'],e['writes'],e['published_rows']) for e in b['events']]
    wa=sum(e['branch_tests'] for e in a['events']); wb=sum(e['branch_tests'] for e in b['events']); assert wb<=wa
    pairs.append({'stream':stream,'fee_cap':fee,'write_cap':write,'WCR_branch_tests':wa,'WQCR_branch_tests':wb,'reduction':wa/wb if wb else None,'same_paid_trace':True})

out={'time':datetime.now().astimezone().isoformat(),'status':'PASS_P_FPB_FRESH_INFERENCE_SQLITE_AND_TRACE_EQUIVALENCE','dataset_rows':96,'actual_predictions':load(D/'inference_receipt.json')['actual_predictions'],'predictor_accuracy_relative_to_supplied_labels':accuracy,'trajectories':len(trajs),'events':event_count,'paid_actions':actions,'sqlite_reopened':len(trajs),'sqlite_errors':db_errors,'unsafe_published_answers':unsafe,'budget_violations':budget_violations,'matched_WCR_WQCR_pairs':len(pairs),'pair_results':pairs,'input_files_reopened':len(load(D/'input_manifest.json')),'source_files_reopened':len(load(D/'source_manifest.json')),'AI_REVIEW':True,'human_reviewed':False,'formal_submission':False,'limits':['Fresh standard-corpus filter execution with actual local model predictions.','Dataset labels are supplied facts, not independently re-annotated semantic truth.','Dynamic updates and fees are synthetic.','No financial join, production concurrency, external engine, or global-novelty certification.']}
assert not db_errors and unsafe==0 and budget_violations==0 and len(pairs)==18
save(OUT/'VERIFICATION.json',out); save(REP/'EXECUTION_VERIFICATION.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='pair_results'},ensure_ascii=False,indent=2))

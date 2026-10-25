"""J independent dictionary/pair reconstruction;separate finite class/bound fixtures."""
from pathlib import Path
from datetime import datetime
import collections,hashlib,itertools,json,random,sqlite3
R=Path(__file__).resolve().parents[1];D=R/'raw_results/materialized_j_v1';A=R/'reports/current_revision';V=R/'derived/materialized_j_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
def digest(x):return hashlib.sha256(json.dumps(sorted(x),separators=(',',':')).encode()).hexdigest()
def effective(rows,ctx,rule,facts,pred,last):
 e={}
 for r in rows:
  i=r['reviewId'];k=r['text_sha256'];mask=rule=='positive_and_top' and r['isTopCritic']!='True'
  known=0 if mask else facts.get(k,-1);b=known if known>=0 else pred.get((k,ctx),last.get(k,0));e[i]=(k,r['id'],b,known,int(not mask and known<0 and (k,ctx) not in pred))
 return e
def answer(e,q):
 if q=='filter':return {(i,'') for i,x in e.items() if x[2]}
 return {(i,j) for i,x in e.items() for j,y in e.items() if i!=j and x[1]==y[1] and x[2]==y[2]}
def cert(e,q):
 if q=='filter':
  a=sum(x[3]==1 for x in e.values());p=sum(x[2] for x in e.values());u0=sum(x[3]<0 and x[2]==0 for x in e.values());return (10*a-9*p if p else 0,a-9*u0,p,sum(x[4] for x in e.values()))
 mp=mr=P=0
 for g in {x[1] for x in e.values()}:
  xs=[x for x in e.values() if x[1]==g];n=len(xs);n1=sum(x[2] for x in xs);a=sum(x[3]==1 for x in xs);u0=sum(x[3]<0 and x[2]==0 for x in xs);u1=sum(x[3]<0 and x[2]==1 for x in xs);p=n1*(n1-1)+(n-n1)*(n-n1-1);vals=[]
  for k0 in range(u0+1):
   for k1 in range(a,a+u1+1):
    tp=k0*(k0-1)+(n-n1-k0)*(n-n1-k0-1)+k1*(k1-1)+(n1-k1)*(n1-k1-1);k=k0+k1;qq=k*(k-1)+(n-k)*(n-k-1);vals.append((10*tp-9*p,10*tp-9*qq))
  mp+=min(x[0] for x in vals);mr+=min(x[1] for x in vals);P+=p
 return (mp,mr,P,sum(x[4] for x in e.values()))
def qualified(c):return c[0]>=0 and c[1]>=0 and c[3]==0
def degree(e,q,i):return 1 if q=='filter' else 2*(sum(x[1]==e[i][1] for x in e.values())-1)
def reserve(e,q,pub):return len(answer(e,q)^pub)+2*sum(degree(e,q,i) for i,x in e.items() if x[3]<0)

def trajectories():
 assert not V.exists();cfg=load(D/'protocol.json');H=R/'raw_results/live_h_v1';streams={x['id']:x for x in load(H/'streams.json')};contexts=load(H/'contexts.json');gold=load(H/'labels_EVALUATION_ONLY.json');pool=load(D/'prediction_pool.json');allrows=[];counts=collections.Counter();pairs={};dbs=[]
 for x in load(D/'input_manifest.json')+load(D/'source_manifest.json'):assert sha(R/x['path'])==x['sha256']
 calls=[json.loads(x) for x in (D/'new_GPU_calls.jsonl').read_text(encoding='utf-8').splitlines()];assert len(calls)==92 and all(x['actual_GPU_forward'] and x['value']==int(x['Yes_No_logits'][0]>=x['Yes_No_logits'][1]) and pool[x['key']]==x['value'] for x in calls)
 for p in sorted((D/'trajectories').glob('*.json')):
  t=load(p);facts={};pred={};last={};candidate=set();pub=set();last_pub=-1;last_cand=-1;bf=t['fee_cap'];bw=t['write_cap'];q=t['query'];history=[]
  for ev,source in zip(t['events'],streams[t['stream']]['events']):
   ctx=contexts[source['context_alias']];e=effective(source['rows'],ctx,source['rule'],facts,pred,last);out=answer(e,q);c=cert(e,q);need=len(candidate^out);accepted=need<=bw;fee=0.;writes=need if accepted else 0
   if accepted:candidate=out;last_cand=source['epoch']
   U=sum(x[3]<0 for x in e.values());Vv=sum(degree(e,q,i) for i,x in e.items() if x[3]<0);fp=bf>=U;jp=t['method'] in {'WCR','WQCR'} and fp and accepted and writes+reserve(e,q,pub)<=bw
   assert ev['initial']=={'cert':list(c),'U':U,'V':Vv,'delta':len(out^pub),'intake_required':need,'intake_applied':accepted,'intake_writes':writes,'joint_protected':jp,'fee_protected':fp}
   for step in ev['steps']:
    assert accepted;i=step['rid'];k=step['key'];kind=step['kind'];assert e[i][0]==k and e[i][3]<0 and (kind=='audit' or e[i][4]);assert step['cert_before']==list(c)
    price=1. if kind=='audit' else .05;assert fee+price<=bf+1e-8
    branches=[]
    for bit in (0,1):
     other=dict(e);v=e[i];other[i]=(v[0],v[1],bit,bit if kind=='audit' else -1,0);o=answer(other,q);cc=cert(other,q);dw=len(out^o);assert dw==(degree(e,q,i) if bit!=v[2] else 0);branches.append((other,o,cc,dw))
    assert max(b[3] for b in branches)<=bw-writes
    terminal=t['method'] in {'CR','WCR','WQCR'} and all(qualified(b[2]) for b in branches)
    if t['method'] in {'WCR','WQCR'}:terminal=terminal and all(writes+b[3]+len(b[1]^pub)<=bw for b in branches)
    if not terminal:
     if fp:assert all(fee+price+sum(x[3]<0 for x in b[0].values())<=bf+1e-8 for b in branches)
     if jp and t['method'] in {'WCR','WQCR'}:assert all(writes+b[3]+reserve(b[0],q,pub)<=bw for b in branches)
    actual=gold[k] if kind=='audit' else pool[k+'|'+ctx];assert actual==step['bit']
    if kind=='audit':assert k not in facts;facts[k]=actual
    else:assert (k,ctx) not in pred;pred[k,ctx]=actual;last[k]=actual
    e=effective(source['rows'],ctx,source['rule'],facts,pred,last);newout=answer(e,q);dw=len(out^newout);writes+=dw;fee+=price;c=cert(e,q);candidate=newout;out=newout
    assert step['candidate_writes']==dw and abs(step['fee_total']-fee)<1e-8 and step['write_total']==writes and step['cert_after']==list(c) and step['delta']==len(out^pub) and step['U']==sum(x[3]<0 for x in e.values()) and step['V']==sum(degree(e,q,i) for i,x in e.items() if x[3]<0) and step['answer_sha256']==digest(out)
    counts['paid_actions']+=1;counts['SQL_effective_states']+=1
   canpublish=accepted and qualified(c) and writes+len(out^pub)<=bw
   if canpublish:
    dw=len(out^pub);pub=out;last_pub=source['epoch'];writes+=dw
    truth={i:(x[0],x[1],int(gold[x[0]]) if source['rule']!='positive_and_top' or next(r for r in source['rows'] if r['reviewId']==i)['isTopCritic']=='True' else 0,0,0) for i,x in e.items()};correct=answer(truth,q);tp=len(out&correct);assert not out or 10*tp>=9*len(out);assert not correct or 10*tp>=9*len(correct);counts['published']+=1
   else:dw=0;counts['refused']+=1
   assert ev['published']==canpublish and ev['publication_writes']==dw and abs(ev['fee']-fee)<1e-8 and ev['writes']==writes<=bw and fee<=bf+1e-8 and ev['final_cert']==list(c) and ev['answer_sha256']==digest(out) and ev['published_table_sha256']==digest(pub)
   if jp and t['method'] in {'WCR','WQCR'}:assert canpublish;counts['joint_protected_events']+=1
   assert len(ev['steps'])<=2*len(e);counts['events']+=1;counts['SQL_effective_states']+=1;history.append([(x['rid'],x['kind'],x['bit']) for x in ev['steps']])
  db=R/t['database'];assert sha(db)==t['database_sha256'];cdb=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);assert cdb.execute('PRAGMA integrity_check').fetchall()==[('ok',)];assert set(cdb.execute('SELECT a,b FROM candidate'))==candidate and set(cdb.execute('SELECT a,b FROM published'))==pub;assert cdb.execute('SELECT * FROM version').fetchone()==(source['epoch'],last_cand,last_pub);sql_e={i:(g,z,k,d) for i,g,z,k,d in cdb.execute('SELECT * FROM effective')};assert sql_e=={i:x[1:] for i,x in e.items()};cdb.close();dbs.append({'path':t['database'],'sha256':sha(db)})
  if t['method'] in {'WCR','WQCR'}:pairs[(t['stream'],q,bf,bw,t['method'])]=(history,[(x['published'],x['fee'],x['writes'],x['published_table_sha256']) for x in t['events']])
  allrows.append({'id':t['id'],'stream':t['stream'],'query':q,'method':t['method'],'fee_cap':bf,'write_cap':bw,'nonempty':sum(x['nonempty'] for x in t['events']),'served':sum(x['nonempty'] and x['published'] for x in t['events']),'fee':sum(x['fee'] for x in t['events']),'writes':sum(x['writes'] for x in t['events']),'steps':sum(len(x['steps']) for x in t['events']),'branch_tests':sum(x['branch_tests'] for x in t['events'])})
 assert len(allrows)==180
 for st,q,bf,bw,method in list(pairs):
  if method=='WCR':assert pairs[st,q,bf,bw,'WCR']==pairs[st,q,bf,bw,'WQCR'];counts['WCR_WQCR_matched_trajectory_pairs']+=1
 groups=collections.defaultdict(list)
 for row in allrows:groups[row['query'],row['fee_cap'],row['write_cap'],row['method']].append(row)
 aggregate=[]
 for (q,bf,bw,m),rows in sorted(groups.items()):
  assert len(rows)==3 and sum(x['nonempty'] for x in rows)==24
  aggregate.append({'query':q,'fee_cap':bf,'write_cap':bw,'method':m,'streams':3,'nonempty_events':24,'served':sum(x['served'] for x in rows),'service_percent':100*sum(x['served'] for x in rows)/24,'mean_stream_fee':sum(x['fee'] for x in rows)/3,'mean_stream_writes':sum(x['writes'] for x in rows)/3,'branch_tests':sum(x['branch_tests'] for x in rows)})
 save(V/'aggregate.json',aggregate);save(V/'trajectories.json',allrows)
 return counts,dbs

def fixtures():
 from d03_materialized_j import State,choose
 rng=random.Random(2026101903);counts=collections.Counter()
 for z in range(512):
  n=rng.randrange(1,5);q=['filter','join'][z%2];rows=[{'reviewId':str(i),'text_sha256':'k'+str(i),'id':'g'+str(rng.randrange(2)),'isTopCritic':'True'} for i in range(n)];facts={};pred={};last={};old={}
  for r in rows:
   k=r['text_sha256'];b=rng.randrange(2);old[r['reviewId']]=(k,'g'+str(rng.randrange(2)),rng.randrange(2),0,0);typ=rng.randrange(3)
   if typ==0:facts[k]=b
   elif typ==1:pred[k,'ctx']=b
   else:last[k]=b
  pub=answer(old,q);s=State(q,rows,'ctx','positive',facts,pred,last,pub);e=effective(rows,'ctx','positive',facts,pred,last);out=answer(e,q);assert s.answer==out and s.cert==cert(e,q);classes={}
  for a in s.actions():
   results=[]
   for b in (0,1):
    branch=s.branch(a,b);v=e[a[0]];ee=dict(e);ee[a[0]]=(v[0],v[1],b,b if a[1]=='audit' else -1,0);aa=answer(ee,q);assert tuple(branch['cert'])==cert(ee,q) and branch['w']==len(out^aa) and branch['delta']==len(aa^pub) and branch['reserve_w']==reserve(ee,q,pub);results.append(branch);counts['explicit_branch_checks']+=1
   k=s.key(a)
   if k in classes:assert classes[k]==results;counts['equal_augmented_class_pairs']+=1
   else:classes[k]=results
  unknown=[i for i,x in e.items() if x[3]<0]
  for bits in itertools.product((0,1),repeat=len(unknown)):
   truth=dict(e)
   for i,b in zip(unknown,bits):v=truth[i];truth[i]=(v[0],v[1],b,b,0)
   aa=answer(truth,q);tp=len(out&aa);assert (10*tp-9*len(out))>=s.cert[0] and (10*tp-9*len(aa))>=s.cert[1]
   for order in [unknown,list(reversed(unknown))]:
    cur=dict(e);w=0;cc=out
    for i in order:
     cur[i]=truth[i];nxt=answer(cur,q);w+=len(cc^nxt);cc=nxt
    assert w+len(cc^pub)<=reserve(e,q,pub);counts['full_audit_paths']+=1
  for bf,bw in [(n,999),(n+.1,s.delta+2*s.V),(.5,max(0,s.delta))]:
   fp=bf>=s.U;jp=fp and bw>=s.delta+2*s.V;assert choose(s,'WCR',bf,bw,fp,jp)[0]==choose(s,'WQCR',bf,bw,fp,jp)[0];counts['fixture_full_quotient_choices']+=1
  counts['fixtures']+=1
 rows=[{'reviewId':str(i),'text_sha256':'k'+str(i),'id':'A' if i<2 else 'B','isTopCritic':'True'} for i in range(4)];facts={'k1':0,'k3':0};last={'k0':0,'k2':0};pub={('0','1'),('1','0')};s=State('join',rows,'ctx','positive',facts,{},last,pub)
 a=('0','refresh');b=('2','refresh');assert s.key(a)[:-2]==s.key(b)[:-2] and s.key(a)!=s.key(b);ra=max(s.branch(a,x)['w']+s.branch(a,x)['reserve_w'] for x in (0,1));rb=max(s.branch(b,x)['w']+s.branch(b,x)['reserve_w'] for x in (0,1));assert (ra,rb)==(14,10) and s.delta+2*s.V==10
 save(V/'counterexample.json',{'query':'join','groups':[['0','1'],['2','3']],'known_zero':['1','3'],'dirty_unknown_zero':['0','2'],'old_published_pairs':sorted(pub),'same_old_quality_class':True,'different_augmented_classes':True,'current_delta':s.delta,'initial_write_reserve':10,'write_remaining':10,'refresh0_worst_step_plus_reserve':ra,'refresh2_worst_step_plus_reserve':rb,'feasible':[False,True]})
 return dict(counts)


# Artifact entry point: separate finite fixtures from archive verification.
V = R / 'derived/j_fixtures_v1'
if V.exists():
    raise RuntimeError('Move prior fixture output before rerunning.')
result = fixtures()
expected = {'explicit_branch_checks': 2570, 'full_audit_paths': 4248,
            'fixture_full_quotient_choices': 1536, 'fixtures': 512,
            'equal_augmented_class_pairs': 70}
assert result == expected, (result, expected)
save(V / 'VERIFICATION.json', {'status': 'PASS_J_FINITE_FIXTURES', 'counts': result})
print(json.dumps({'status': 'PASS_J_FINITE_FIXTURES', 'counts': result}, indent=2))
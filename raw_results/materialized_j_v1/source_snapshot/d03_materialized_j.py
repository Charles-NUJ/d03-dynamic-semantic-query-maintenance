"""D03 J: singleton acquisition with paid-fee and result-membership-write budgets."""
import collections,hashlib,json,math,sqlite3

def digest(rows):return hashlib.sha256(json.dumps(sorted(rows),separators=(',',':')).encode()).hexdigest()
def F(n,k):return k*(k-1)+(n-k)*(n-k-1)
def local(query,t):
 n,n1,a,u0,u1,d=t
 if query=='filter':return (10*a-9*n1 if n1 else 0,a-9*u0,n1,d)
 n0=n-n1;p=F(n,n1)
 def low(n,l,h):
  ks={l,h,max(l,min(h,n//2)),max(l,min(h,(n+1)//2))};return min(F(n,k) for k in ks)
 tp=low(n0,0,u0)+low(n1,a,a+u1);mr=math.inf
 for k0 in range(u0+1):
  v=(36*k0+20*n1-18*n)/4
  for k1 in {max(a,min(a+u1,math.floor(v))),max(a,min(a+u1,math.ceil(v)))}:mr=min(mr,10*(F(n0,k0)+F(n1,k1))-9*F(n,k0+k1))
 return (10*tp-9*p if p else 0,int(mr),p,d)

class State:
 def __init__(self,query,rows,ctx,rule,facts,pred,last,published):
  self.query=query;self.ctx=ctx;self.rule=rule;self.original={r['reviewId']:r for r in rows};self.e={}
  for r in sorted(rows,key=lambda x:x['reviewId']):
   key=r['text_sha256'];mask=rule=='positive_and_top' and r['isTopCritic']!='True'
   known=0 if mask else facts.get(key,-1);z=known if known>=0 else pred.get((key,ctx),last.get(key,0));dirty=int(not mask and known<0 and (key,ctx) not in pred)
   self.e[r['reviewId']]=(key,r['id'],z,known,dirty)
  self.published=set(published);self.rebuild()
 def rebuild(self):
  self.groups=collections.defaultdict(list)
  for i,x in self.e.items():self.groups[x[1]].append(i)
  self.counts={}
  for g,ids in self.groups.items():
   xs=[self.e[i] for i in ids];self.counts[g]=(len(xs),sum(x[2] for x in xs),sum(x[3]==1 for x in xs),sum(x[3]<0 and x[2]==0 for x in xs),sum(x[3]<0 and x[2]==1 for x in xs),sum(x[4] for x in xs))
  ls=[local(self.query,t) for t in self.counts.values()];self.cert=tuple(sum(v[k] for v in ls) for k in range(4));self.answer=self.outputs();self.diff=self.answer^self.published;self.delta=len(self.diff);self.U=sum(x[3]<0 for x in self.e.values());self.V=sum(self.degree(i) for i,x in self.e.items() if x[3]<0)
 def outputs(self):
  if self.query=='filter':return {(i,'') for i,x in self.e.items() if x[2]}
  return {(i,j) for ids in self.groups.values() for i in ids for j in ids if i!=j and self.e[i][2]==self.e[j][2]}
 def qualified(self):return self.cert[0]>=0 and self.cert[1]>=0 and self.cert[3]==0
 def degree(self,i):return 1 if self.query=='filter' else 2*(len(self.groups[self.e[i][1]])-1)
 def crossing(self,i):
  if self.query=='filter':return int((i,'') in self.diff)
  return sum(int((i,j) in self.diff)+int((j,i) in self.diff) for j in self.groups[self.e[i][1]] if j!=i)
 def actions(self):
  out=[]
  for i,x in self.e.items():
   if x[3]>=0:continue
   out.append((i,'audit'))
   if x[4]:out.append((i,'refresh'))
  return out
 def key(self,a):
  i,kind=a;x=self.e[i];return (self.query,self.counts[x[1]],kind,x[2],x[4],self.degree(i),self.crossing(i))
 def branch(self,a,b):
  i,kind=a;key,g,z,known,dirty=self.e[i];t=list(self.counts[g]);t[1]+=b-z;t[5]-=dirty
  t[3+z]-=1
  if kind=='audit':t[2]+=b
  else:t[3+b]+=1
  old=local(self.query,self.counts[g]);new=local(self.query,tuple(t));cert=tuple(self.cert[k]-old[k]+new[k] for k in range(4));D=self.degree(i);w=D if b!=z else 0;d=self.delta+(D-2*self.crossing(i) if b!=z else 0);u=self.U-int(kind=='audit');v=self.V-(D if kind=='audit' else 0)
  return {'cert':cert,'qualified':cert[0]>=0 and cert[1]>=0 and cert[3]==0,'w':w,'delta':d,'U':u,'V':v,'reserve_w':d+2*v}
 def apply(self,a,b):
  i,kind=a;key,g,z,k,d=self.e[i];self.e[i]=(key,g,b,b if kind=='audit' else -1,0);self.rebuild()

def choose(s,method,fee_left,write_left,fee_protected,joint_protected,cm=.05,ca=1.):
 actions=s.actions();reps=[];seen=set()
 for a in actions:
  k=s.key(a)
  if method=='WQCR' and k in seen:continue
  seen.add(k);reps.append(a)
 candidates=[];tests=0;aware=method in {'WCR','WQCR'}
 for order,a in enumerate(reps):
  fee=ca if a[1]=='audit' else cm
  if fee>fee_left+1e-9:continue
  bs=[s.branch(a,b) for b in (0,1)];tests+=2
  if max(x['w'] for x in bs)>write_left:continue
  terminal=method in {'CR','WCR','WQCR'} and all(x['qualified'] for x in bs)
  if aware:terminal=terminal and all(x['w']+x['delta']<=write_left for x in bs)
  if not terminal:
   if fee_protected and any(fee+ca*x['U']>fee_left+1e-9 for x in bs):continue
   if aware and joint_protected and any(x['w']+x['reserve_w']>write_left for x in bs):continue
  risk=max(x['w']+x['reserve_w'] for x in bs);commit=max(x['w']+x['delta'] for x in bs)
  candidates.append((a,fee,terminal,risk,commit,order))
 terminals=[x for x in candidates if x[2]]
 if terminals:
  q=min(terminals,key=lambda x:(x[1],x[4] if aware else 0,x[5]));return q[0],tests,len(reps)
 audits=[x for x in candidates if x[0][1]=='audit'];refresh=[x for x in candidates if x[0][1]=='refresh'];quality=s.cert[0]>=0 and s.cert[1]>=0
 target=refresh if refresh and (method=='SC' or (quality and cm<ca)) else audits if audits else refresh
 if not target:return None,tests,len(reps)
 q=min(target,key=lambda x:(x[3],x[5])) if aware else target[0]
 return q[0],tests,len(reps)

class SQL:
 """Explicit candidate and published result tables;cache/metadata writes are excluded units."""
 def __init__(self,path,query):
  self.query=query;self.c=sqlite3.connect(path,isolation_level=None)
  self.c.executescript('CREATE TABLE effective(rid TEXT PRIMARY KEY,g TEXT,z INT,known INT,dirty INT);CREATE TABLE candidate(a TEXT,b TEXT,PRIMARY KEY(a,b));CREATE TABLE published(a TEXT,b TEXT,PRIMARY KEY(a,b));CREATE TABLE version(current INT,candidate INT,published INT);INSERT INTO version VALUES(-1,-1,-1);')
 def rows(self,t):return set(self.c.execute('SELECT a,b FROM '+t))
 def full(self):
  q="SELECT rid,'' FROM effective WHERE z=1" if self.query=='filter' else 'SELECT x.rid,y.rid FROM effective x JOIN effective y ON x.g=y.g AND x.z=y.z AND x.rid<>y.rid'
  return set(self.c.execute(q))
 def sync_effective(self,s):
  self.c.execute('DELETE FROM effective');self.c.executemany('INSERT INTO effective VALUES(?,?,?,?,?)',[(i,x[1],x[2],x[3],x[4]) for i,x in s.e.items()]);assert self.full()==s.answer
 def delta_write(self,t,wanted):
  old=self.rows(t);delete=old-wanted;insert=wanted-old
  self.c.executemany('DELETE FROM '+t+' WHERE a=? AND b=?',sorted(delete));self.c.executemany('INSERT INTO '+t+' VALUES(?,?)',sorted(insert));assert self.rows(t)==wanted;return len(delete)+len(insert)
 def intake(self,s,epoch,cap):
  self.c.execute('BEGIN IMMEDIATE');self.sync_effective(s);self.c.execute('UPDATE version SET current=?',(epoch,));cost=len(self.rows('candidate')^s.answer)
  if cost>cap:self.c.execute('COMMIT');return False,0,cost
  w=self.delta_write('candidate',s.answer);self.c.execute('UPDATE version SET candidate=?',(epoch,));self.c.execute('COMMIT');return True,w,cost
 def paid(self,s,expected_w):
  self.c.execute('BEGIN IMMEDIATE');self.sync_effective(s);w=self.delta_write('candidate',s.answer);assert w==expected_w;self.c.execute('COMMIT');return w
 def publish(self,s,epoch,remaining):
  assert s.qualified() and self.rows('candidate')==s.answer;w=len(self.rows('published')^s.answer)
  if w>remaining:return False,0
  self.c.execute('BEGIN IMMEDIATE');actual=self.delta_write('published',s.answer);assert actual==w;self.c.execute('UPDATE version SET published=?',(epoch,));self.c.execute('COMMIT');return True,w
 def read(self,epoch):
  cur,cand,pub=self.c.execute('SELECT * FROM version').fetchone();return self.rows('published') if cur==cand==pub==epoch else None

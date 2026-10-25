"""D03 K:safe-class physical QP,paid-only ridge feedback,and linear-ranking ablation."""
import hashlib,json
import numpy as np
from d03_materialized_j import State,SQL,digest,choose as baseline_choose

TAU=.01
WEIGHTS=np.array([1.,1.,.25,.1,.25])
FEE_WEIGHT=.005
RIDGE=1.
FORGET=.98
TOL=1.e-8
MAX_ITER=1200

def checksum(a,b):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()+np.asarray(b,dtype='<f8').tobytes()).hexdigest()
def simplex(v):
 u=np.sort(v)[::-1];c=np.cumsum(u)-1.;j=np.arange(1,len(v)+1);ok=u-c/j>0;rho=np.flatnonzero(ok)[-1];return np.maximum(v-c[rho]/(rho+1),0.)

def feature(s,a):
 i,kind=a;x=s.e[i];n,n1,known,u0,u1,dirty=s.counts[x[1]];D=s.degree(i)
 return np.array([1.,float(x[2]),float(x[4]),n/max(1,len(s.e)),(u0+u1)/max(1,n),s.crossing(i)/max(1,D),D/max(1,2*(len(s.e)-1))])

class Learner:
 def __init__(self):self.models={}
 def model(self,s,a):
  key=(a[1],s.ctx if a[1]=='refresh' else 'fixed_predicate')
  if key not in self.models:
   prior=np.zeros(7);prior[0]=.5;self.models[key]=[RIDGE*np.eye(7),RIDGE*prior,prior,0]
  return self.models[key]
 def predict(self,s,a):
  A,b,prior,n=self.model(s,a);theta=np.linalg.solve(A,b);return float(np.clip(feature(s,a)@theta,0,1))
 def update(self,s,a,bit):
  A,b,prior,n=self.model(s,a);phi=feature(s,a);p=self.predict(s,a);target=int(bit!=s.e[a[0]][2]);before=checksum(A,b)
  An=FORGET*A+(1-FORGET)*RIDGE*np.eye(7)+np.outer(phi,phi);bn=FORGET*b+(1-FORGET)*RIDGE*prior+phi*target
  key=(a[1],s.ctx if a[1]=='refresh' else 'fixed_predicate');self.models[key]=[An,bn,prior,n+1]
  return {'probability_before':p,'target_flip':target,'squared_error':(p-target)**2,'samples_before':n,'features':phi.tolist(),'before_sha256':before,'after_sha256':checksum(An,bn)}

def eligible(s,rf,rw,fp,jp,cm=.05,ca=1.):
 out=[];seen=set();tests=0
 for order,a in enumerate(s.actions()):
  key=s.key(a)
  if key in seen:continue
  seen.add(key);fee=ca if a[1]=='audit' else cm
  if fee>rf+1e-9:continue
  bs=[s.branch(a,b) for b in (0,1)];tests+=2
  if max(x['w'] for x in bs)>rw:continue
  terminal=all(x['qualified'] and x['w']+x['delta']<=rw for x in bs)
  if not terminal:
   if fp and any(fee+ca*x['U']>rf+1e-9 for x in bs):continue
   if jp and any(x['w']+x['reserve_w']>rw for x in bs):continue
  out.append({'action':a,'fee':fee,'branches':bs,'terminal':terminal,'order':order,'commit':max(x['w']+x['delta'] for x in bs)})
 return out,tests

def planning(s,candidates,probs,rf,rw,bf,bw):
 N=max(1,len(s.e));sq=max(1.,10.*len(s.e),abs(s.cert[0]),abs(s.cert[1]));sf=max(1.,bf);sw=max(1.,bw);kf=rf-s.U;kw=rw-s.delta-2*s.V
 target=np.array([max(0.,-s.cert[0])/sq,max(0.,-s.cert[1])/sq,s.cert[3]/N,max(0.,-kf)/sf,max(0.,.1-kw/sw)])
 columns=[]
 for c,p in zip(candidates,probs):
  bs=c['branches'];vectors=[]
  for b in bs:
   vectors.append(np.array([(b['cert'][0]-s.cert[0])/sq,(b['cert'][1]-s.cert[1])/sq,(s.cert[3]-b['cert'][3])/N,(-c['fee']-(b['U']-s.U))/sf,(-b['w']-(b['reserve_w']-s.delta-2*s.V))/sw]))
  z=s.e[c['action'][0]][2];columns.append((1-p)*vectors[z]+p*vectors[1-z])
 G=np.array(columns).T;cost=FEE_WEIGHT*np.array([c['fee'] for c in candidates]);return G,target,cost

def objective(G,target,cost,x):
 v=G@x-target;return float(.5*np.dot(WEIGHTS*v,v)+TAU*.5*(x@x)+cost@x)

def solve(G,target,cost):
 n=G.shape[1];x=np.full(n,1./n);L=float(np.sum(WEIGHTS[:,None]*G*G)+TAU);assert L>=TAU>0;initial=objective(G,target,cost,x);it=0
 for it in range(1,MAX_ITER+1):
  grad=G.T@(WEIGHTS*(G@x-target))+TAU*x+cost;y=simplex(x-grad/L);res=float(L*np.linalg.norm(x-y));x=y
  if res<=TOL:break
 grad=G.T@(WEIGHTS*(G@x-target))+TAU*x+cost;res=float(L*np.linalg.norm(x-simplex(x-grad/L)));f=objective(G,target,cost,x);assert f<=initial+1e-10
 return x,{'iterations':it,'L_upper':L,'gradient_mapping_norm':res,'converged':res<=TOL,'objective_initial':initial,'objective_final':f}

def choose(s,method,rf,rw,fp,jp,bf,bw,learner):
 if method=='WQCR':
  a,tests,reps=baseline_choose(s,method,rf,rw,fp,jp);return a,{'mode':'WQCR','branch_tests':tests,'representatives':reps}
 candidates,tests=eligible(s,rf,rw,fp,jp)
 if not candidates:return None,{'mode':'NO_ADMISSIBLE_ACTION','branch_tests':tests,'classes':0}
 terminals=[c for c in candidates if c['terminal']]
 if terminals:
  c=min(terminals,key=lambda c:(c['fee'],c['commit'],c['order']));return c['action'],{'mode':'TERMINAL','branch_tests':tests,'classes':len(candidates)}
 probs=[.5 if method=='SQP' else learner.predict(s,c['action']) for c in candidates];G,b,cost=planning(s,candidates,probs,rf,rw,bf,bw)
 if method=='LRANK':
  score=cost-G.T@(WEIGHTS*b);j=int(np.argmin(score));info={'mode':'LINEAR_RANK','scores':score.tolist()}
 else:
  x,info=solve(G,b,cost);j=int(np.argmax(x));info.update(mode='QP',allocation=x.tolist())
 info.update(branch_tests=tests,classes=len(candidates),candidate_actions=[list(c['action']) for c in candidates],probabilities=probs,selected_class=j)
 return candidates[j]['action'],info

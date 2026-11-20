import json,math
# curated from bake_lanes.py detection (asphalt grid) + CityLayout.ROADS east extension (x 6..137, vertical x=71.5)
H=[ # z, x0, x1, width
(-172.3,-235.3,137.0,11.5),
(-143.6,-235.3,-195.3,11.0),(-143.6,-177.8,-108.8,11.0),(-143.6,-91.3,-36.8,11.0),(-143.6,6.0,137.0,11.0),
(-114.8,-235.3,-22.3,11.5),(-114.8,21.0,137.0,11.0),
(-100.3,-307.3,-223.8,11.5),
(-86.1,-4.8,64.2,11.0),
(-57.1,-307.3,137.0,11.0),
(-28.3,-206.3,-108.8,11.5),(-28.1,-62.3,-22.3,11.0),(-28.3,6.0,137.0,11.0),
(0.4,-235.3,-79.8,11.0),(0.4,6.0,137.0,11.0)]
V=[ # x, z0, z1, width
(-301.6,-106.1,-37.1,11.5),(-229.6,-178.1,5.9,11.5),
(-200.8,-178.1,-138.1,11.0),(-200.8,-34.1,5.9,11.0),
(-172.0,-178.1,-80.6,11.5),(-171.9,-62.6,5.9,11.0),
(-143.3,-149.1,-109.1,11.0),(-114.3,-178.1,5.9,11.0),(-85.6,-178.1,-51.6,11.5),
(-56.8,-120.6,5.9,11.0),(-42.3,-178.1,-109.1,11.0),(-27.8,-120.6,-94.6,11.0),
(-29.1,-62.6,-22.6,8.5),(0.7,-178.1,5.9,11.0),(71.5,-178.1,5.9,11.0)]

def snapH(z,x0,x1,w):
    def f(p):
        for x,za,zb,wv in V:
            if abs(p-x)<=wv/2+1.5 and za-6<=z<=zb+6: return x
        return p
    return (z,f(x0),f(x1),w)
def snapV(x,z0,z1,w):
    def f(p):
        for z,xa,xb,wh in H:
            if abs(p-z)<=wh/2+1.5 and xa-6<=x<=xb+6: return z
        return p
    return (x,f(z0),f(z1),w)
H=[snapH(*h) for h in H]; V=[snapV(*v) for v in V]
TOL=1.0
nodes={}   # key -> idx
NL=[]      # [x,z,hwh,hwv]
def node(x,z):
    for i,n in enumerate(NL):
        if abs(n[0]-x)<TOL*1.5 and abs(n[1]-z)<TOL*1.5: return i
    NL.append([x,z,0.0,0.0]); return len(NL)-1
edges=set(); E=[]
for z,x0,x1,w in H:
    pts=[x0,x1]
    for x,za,zb,wv in V:
        if x0-TOL<=x<=x1+TOL and za-TOL<=z<=zb+TOL: pts.append(x)
    pts=sorted(set(round(p,1) for p in pts)); ids=[]
    for p in pts:
        i=node(p,z); ids.append(i)
        if p not in (round(x0,1),round(x1,1)) or any(abs(p-x)<TOL and za-TOL<=z<=zb+TOL for x,za,zb,wv in V): NL[i][2]=max(NL[i][2],w/2)
        else: NL[i][2]=max(NL[i][2],w/2)
    for a,b in zip(ids,ids[1:]):
        if a!=b and (min(a,b),max(a,b)) not in edges: edges.add((min(a,b),max(a,b))); E.append((a,b,w))
for x,z0,z1,w in V:
    pts=[z0,z1]
    for z,xa,xb,wh in H:
        if z0-TOL<=z<=z1+TOL and xa-TOL<=x<=xb+TOL: pts.append(z)
    pts=sorted(set(round(p,1) for p in pts)); ids=[]
    for p in pts:
        i=node(x,p); ids.append(i); NL[i][3]=max(NL[i][3],w/2)
    for a,b in zip(ids,ids[1:]):
        if a!=b and (min(a,b),max(a,b)) not in edges: edges.add((min(a,b),max(a,b))); E.append((a,b,w))
deg=[0]*len(NL)
for a,b,w in E: deg[a]+=1; deg[b]+=1
short=[(a,b,math.dist(NL[a][:2],NL[b][:2])) for a,b,w in E if math.dist(NL[a][:2],NL[b][:2])<12]
print('nodes',len(NL),'edges',len(E),'intersections(deg>=3)',sum(d>=3 for d in deg),'deadends',sum(d==1 for d in deg),'short<12m',len(short))
for s in short: print(' short',s)
json.dump({'nodes':NL,'edges':E,'deg':deg},open('graph.json','w'))

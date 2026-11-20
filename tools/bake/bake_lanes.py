import numpy as np, json
g=np.load('/home/claude/grid.npy');A=(g==1)
ox,oz,C=-308.8,-179.6,0.5
MINRUN=40  # cells (20 m) of continuous asphalt = a street, not a crossing
def bands(M):
    # M[r,c] asphalt; horizontal streets along c. returns (r0,r1,c0,c1)
    H,W=M.shape; rows={}
    for r in range(H):
        row=M[r]; c=0
        while c<W:
            if row[c]:
                s=c
                while c<W and row[c]: c+=1
                if c-s>=MINRUN: rows.setdefault(r,[]).append((s,c-1))
            else: c+=1
    out=[]; active=[]  # (r0,r1,c0,c1)
    for r in sorted(rows):
        for (s,e) in rows[r]:
            for b in active:
                if b[1]==r-1 and not(e<b[2] or s>b[3]):
                    b[1]=r; b[2]=min(b[2],s); b[3]=max(b[3],e); break
            else: active.append([r,r,s,e])
    return [b for b in active if b[1]-b[0]+1>=14]  # >=7 m wide
hb=bands(A); vb=bands(A.T)   # vb: (c0,c1,r0,r1)
# merge bands with near-identical z ranges (same street split by gaps)
def merge(bs,tol=4):
    bs=sorted(bs,key=lambda b:(b[0],b[2])); out=[]
    for b in bs:
        for o in out:
            if abs(o[0]-b[0])<=tol and abs(o[1]-b[1])<=tol and b[2]<=o[3]+12:
                o[2]=min(o[2],b[2]); o[3]=max(o[3],b[3]); o[0]=min(o[0],b[0]); o[1]=max(o[1],b[1]); break
        else: out.append(list(b))
    return out
hb=merge(hb); vb=merge(vb)
HS=[(oz+(r0+r1+1)/2*C, ox+c0*C, ox+(c1+1)*C, (r1-r0+1)*C) for r0,r1,c0,c1 in hb]  # z, x0,x1,width
VS=[(ox+(c0+c1+1)/2*C, oz+r0*C, oz+(r1+1)*C, (c1-c0+1)*C) for c0,c1,r0,r1 in vb]  # x, z0,z1,width
print(len(HS),'H',len(VS),'V')
for s in HS: print('H z=%.1f x %.1f..%.1f w=%.1f'%s)
for s in VS: print('V x=%.1f z %.1f..%.1f w=%.1f'%s)
json.dump({'H':HS,'V':VS},open('/home/claude/bake/streets.json','w'))

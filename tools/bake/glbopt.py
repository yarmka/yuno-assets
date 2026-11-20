#!/usr/bin/env python3
"""Lossless glb shrinker: (1) uint32 indices -> uint16 when every index < 65536, (2) drops an all-white COLOR_0 attribute
(a no-op multiplier), (3) removes bufferViews/accessors nobody references, re-packs the BIN chunk. Nothing else is touched."""
import struct,json,sys,numpy as np
CT={5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'};NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
def read(p):
    b=open(p,'rb').read();assert b[:4]==b'glTF'
    n=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+n]);o=20+n;bn=b''
    if o<len(b):
        l=struct.unpack('<I',b[o:o+4])[0];bn=b[o+8:o+8+l]
    return j,bn
def acc_data(j,bn,i):
    a=j['accessors'][i];bv=j['bufferViews'][a['bufferView']];off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    nc=NC[a['type']];dt=np.dtype(CT[a['componentType']]);stride=bv.get('byteStride',nc*dt.itemsize)
    if stride==nc*dt.itemsize: return np.frombuffer(bn,dtype=dt,count=a['count']*nc,offset=off).reshape(a['count'],nc)
    raw=np.frombuffer(bn,dtype=np.uint8,offset=off)
    return np.stack([np.frombuffer(raw[k*stride:k*stride+nc*dt.itemsize].tobytes(),dtype=dt) for k in range(a['count'])])
def optimize(j,bn):
    acc=j['accessors'];new={}   # accessor index -> replacement bytes (uint16 indices)
    # which accessors are indices / colors
    idx=set();col={}
    for m in j.get('meshes',[]):
        for pr in m['primitives']:
            if 'indices' in pr: idx.add(pr['indices'])
            if 'COLOR_0' in pr['attributes']: col.setdefault(pr['attributes']['COLOR_0'],[]).append(pr)
    # accessor use counts outside meshes (animations, skins, sparse) -> skip those
    other=set()
    for s in j.get('skins',[]): other.add(s.get('inverseBindMatrices'))
    for an in j.get('animations',[]):
        for sm in an['samplers']: other.add(sm['input']);other.add(sm['output'])
    drop=set()
    for ai,prs in col.items():
        a=acc[ai]
        if ai in other or 'sparse' in a: continue
        d=acc_data(j,bn,ai).astype(np.float64)
        if a.get('normalized') or a['componentType']!=5126: d=d/{5121:255.0,5123:65535.0}.get(a['componentType'],1.0)
        if d.shape[1]>=3 and (d[:,:3]>=0.9999).all() and (d.shape[1]==3 or (d[:,3]>=0.9999).all()):
            for pr in prs: pr['attributes'].pop('COLOR_0',None)
            drop.add(ai)
    for ai in idx:
        a=acc[ai]
        if a['componentType']!=5125 or ai in other or 'sparse' in a: continue
        d=acc_data(j,bn,ai).reshape(-1)
        if d.size and d.max()<65535: new[ai]=d.astype('<u2').tobytes()
    # accessors still referenced?
    used=set();[used.update(pr['attributes'].values()) for m in j.get('meshes',[]) for pr in m['primitives']]
    for m in j.get('meshes',[]):
        for pr in m['primitives']:
            if 'indices' in pr: used.add(pr['indices'])
            for t in pr.get('targets',[]): used.update(t.values())
    used|={x for x in other if x is not None}
    keep=[i for i in range(len(acc)) if i not in drop or i in used]
    amap={o:n for n,o in enumerate(keep)}
    # rebuild buffer: bufferViews referenced by kept accessors / images
    bvs=j['bufferViews'];newbv=[];newbin=bytearray();bvmap={}
    def add(data,extra=None):
        while len(newbin)%4: newbin.append(0)
        e=dict(extra or {});e['buffer']=0;e['byteOffset']=len(newbin);e['byteLength']=len(data);newbin.extend(data);newbv.append(e);return len(newbv)-1
    # accessors with replaced data get their own view
    for ai in keep:
        a=acc[ai]
        if 'bufferView' not in a: continue
        if ai in new:
            a['bufferView']=add(new[ai],{'target':34963});a['componentType']=5123;a.pop('byteOffset',None)
        else:
            ob=a['bufferView']
            if ob not in bvmap:
                bv=bvs[ob];d=bn[bv.get('byteOffset',0):bv.get('byteOffset',0)+bv['byteLength']]
                ex={k:v for k,v in bv.items() if k in('byteStride','target','name')}
                bvmap[ob]=add(d,ex)
            a['bufferView']=bvmap[ob]
    for im in j.get('images',[]):
        if 'bufferView' in im:
            ob=im['bufferView']
            if ob not in bvmap:
                bv=bvs[ob];bvmap[ob]=add(bn[bv.get('byteOffset',0):bv.get('byteOffset',0)+bv['byteLength']])
            im['bufferView']=bvmap[ob]
    # remap accessor indices
    j['accessors']=[acc[i] for i in keep]
    for m in j.get('meshes',[]):
        for pr in m['primitives']:
            pr['attributes']={k:amap[v] for k,v in pr['attributes'].items()}
            if 'indices' in pr: pr['indices']=amap[pr['indices']]
            for t in pr.get('targets',[]):
                for k in t: t[k]=amap[t[k]]
    for s in j.get('skins',[]):
        if s.get('inverseBindMatrices') is not None: s['inverseBindMatrices']=amap[s['inverseBindMatrices']]
    for an in j.get('animations',[]):
        for sm in an['samplers']: sm['input']=amap[sm['input']];sm['output']=amap[sm['output']]
    j['bufferViews']=newbv;j['buffers']=[{'byteLength':len(newbin)}]
    return j,bytes(newbin),len(drop),len(new)
def write(p,j,bn):
    js=json.dumps(j,separators=(',',':')).encode()
    while len(js)%4: js+=b' '
    while len(bn)%4: bn+=b'\0'
    out=b'glTF'+struct.pack('<II',2,12+8+len(js)+8+len(bn))+struct.pack('<I',len(js))+b'JSON'+js+struct.pack('<I',len(bn))+b'BIN\0'+bn
    open(p,'wb').write(out)
if __name__=='__main__':
    src,dst=sys.argv[1:3];j,bn=read(src)
    if 'sparse' in json.dumps(j.get('accessors',[])) or any('uri' in b for b in j.get('buffers',[])): print('skip',src);sys.exit(2)
    before=len(open(src,'rb').read());j2,bn2,nd,ni=optimize(j,bn);write(dst,j2,bn2);after=len(open(dst,'rb').read())
    print(f'{src}: {before/1e3:.0f} -> {after/1e3:.0f} KB (colors dropped {nd}, indices16 {ni})')

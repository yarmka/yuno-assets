# glTF post-fix: Blender's importer makes "Name.001" copies of materials; the game finds materials by their ORIGINAL names -> strip the suffix.
import struct,json,sys,re
p=sys.argv[1]; orig=sys.argv[2]
def _names(f):
    o=open(f,'rb').read(); return {m.get('name') for m in json.loads(o[20:20+struct.unpack('<I',o[12:16])[0]]).get('materials',[])}
ON=_names(orig)
d=open(p,'rb').read()
l=struct.unpack('<I',d[12:16])[0]; j=json.loads(d[20:20+l]); rest=d[20+l:]
n=0
for m in j.get('materials',[]):
    nm=m.get('name','')
    if nm not in ON and re.search(r'\.\d{3}$',nm) and nm[:-4] in ON: m['name']=nm[:-4]; n+=1
js=json.dumps(j,separators=(',',':')).encode()
js+=b' '*((4-len(js)%4)%4)
out=d[:8]+struct.pack('<I',12+8+len(js)+len(rest))+struct.pack('<I',len(js))+b'JSON'+js+rest
open(p,'wb').write(out); print("renamed",n)

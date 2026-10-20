"""Independent DeepStack capacity key hashing, table recovery and lookup.

Only extraction reads the original binary. load() and lookup() need the
recovered data and this pure-Python module; no original extension is imported.
"""
import bisect
import csv
import struct
from pathlib import Path
from .recovered_capacity_key import canonical_words

MASK=(1<<64)-1
COUNT=96825
MAGIC=b'DSCAPLUT1'
def rol(x,n):return ((x<<n)|(x>>(64-n)))&MASK
def sipround(v):
    a,b,c,d=v
    a=(a+b)&MASK;b=rol(b,13)^a;a=rol(a,32)
    c=(c+d)&MASK;d=rol(d,16)^c
    a=(a+d)&MASK;d=rol(d,21)^a
    c=(c+b)&MASK;b=rol(b,17)^c;c=rol(c,32)
    return a,b,c,d
def siphash24(message,k0,k1):
    v=(0x736f6d6570736575^k0,0x646f72616e646f6d^k1,0x6c7967656e657261^k0,0x7465646279746573^k1)
    def block(v,m):
        a,b,c,d=v;v=(a,b,c,d^m)
        v=sipround(sipround(v));a,b,c,d=v
        return a^m,b,c,d
    stop=len(message)//8*8
    for off in range(0,stop,8):v=block(v,struct.unpack_from('<Q',message,off)[0])
    tail=(len(message)&255)<<56
    for i,b in enumerate(message[stop:]):tail|=b<<(8*i)
    v=block(v,tail);a,b,c,d=v;v=(a,b,c^255,d)
    for _ in range(4):v=sipround(v)
    a,b,c,d=v;return a^b^c^d

def hashes(words):
    first=struct.pack('<20Q',0x44534341504c5551,*words)
    second=struct.pack('<20Q',0xda643af82f062944,*words)
    h1=siphash24(first,rol(0xa671e32c5ed248a3,17)^0xe9238bea635e8e99,rol(0xdfac4d0db3d4993e,17)^0x8b84032f8bdd4bcf)
    h2=siphash24(second,rol(0x50ad85e40df9fffb,17)^0xc42d7edb9fe875a7,rol(0x0e3b2865f84425c4,17)^0xcdc63736d7c0b9b4)
    return h1,h2

def finalizer(x):
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def decode(h1,h2,payload,check):
    expected=finalizer(rol(0x3cb9c3faacc055af,17)^0x48ffd8c98d77a0ff^h2^rol(h1,29)^((payload*0xd1342543de82ef95)&MASK))&255
    value=payload^(finalizer(rol(0x0aa37a225f314594,17)^0xc9dfb8d10e4759fc^h1^rol(h2,17))&255)
    if check!=expected or not 1<=value<=128:
        raise ValueError('invalid protected configuration')
    return value

class CapacityLUT:
    def __init__(self,records):
        self.records=list(records);self.keys=[(a,b) for a,b,c in self.records]
        if self.keys!=sorted(self.keys):raise ValueError('LUT order mismatch')
    @classmethod
    def from_provider_file(cls,path):
        data=Path(path).read_bytes()
        h1=struct.unpack_from('<%dQ'%COUNT,data,0xf7e40)
        h2=struct.unpack_from('<%dQ'%COUNT,data,0x3ac60)
        payload=data[0x23220:0x23220+COUNT]
        checks=data[0xb7e0:0xb7e0+COUNT]
        return cls((a,b,decode(a,b,p,c)) for a,b,p,c in zip(h1,h2,payload,checks))
    def save(self,path):
        with Path(path).open('wb') as f:
            f.write(MAGIC);f.write(struct.pack('<I',len(self.records)))
            for row in self.records:f.write(struct.pack('<QQB',*row))
    @classmethod
    def load(cls,path):
        data=Path(path).read_bytes()
        if data[:len(MAGIC)]!=MAGIC:raise ValueError('invalid LUT header')
        count=struct.unpack_from('<I',data,len(MAGIC))[0];start=len(MAGIC)+4
        if len(data)!=start+count*17:raise ValueError('invalid LUT size')
        return cls(struct.iter_unpack('<QQB',data[start:]))
    def write_csv(self,path):
        with Path(path).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['hash1_hex','hash2_hex','capacity'])
            writer.writerows((f'{a:016x}',f'{b:016x}',c) for a,b,c in self.records)
    def lookup_words(self,words):
        key=hashes(words);i=bisect.bisect_left(self.keys,key)
        if i==len(self.keys) or self.keys[i]!=key:raise ValueError('unsupported configuration')
        return self.records[i][2]
    def lookup(self,architecture,hierarchy):return self.lookup_words(canonical_words(architecture,hierarchy))

if __name__=='__main__':
    import argparse,json,collections
    p=argparse.ArgumentParser();p.add_argument('provider');p.add_argument('output');args=p.parse_args()
    lut=CapacityLUT.from_provider_file(args.provider);lut.save(args.output);lut.write_csv(str(Path(args.output).with_suffix('.csv')))
    print(json.dumps({'entries':len(lut.records),'value_histogram':dict(sorted(collections.Counter(r[2] for r in lut.records).items()))},indent=2))

from __future__ import annotations
import gzip, json, struct, urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp

OUT=Path('flyevo/results'); OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('flyevo/flybrain-connectome.bin.gz')
URL='https://raw.githubusercontent.com/snedea/flybrain/main/data/connectome.bin.gz'
META_URL='https://raw.githubusercontent.com/snedea/flybrain/main/data/neuron_meta.json'

if not DATA.exists():
    print('downloading',URL)
    urllib.request.urlretrieve(URL,DATA)
with urllib.request.urlopen(META_URL) as r:
    meta=json.load(r)

raw=gzip.decompress(DATA.read_bytes())
n,e=struct.unpack_from('<II',raw,0)
edge_dtype=np.dtype([('pre','<u4'),('post','<u4'),('w','<f4')])
edges=np.frombuffer(raw,dtype=edge_dtype,count=e,offset=8)
off=8+e*12
node_dtype=np.dtype([('region','u1'),('group','<u2')])
nodes=np.frombuffer(raw,dtype=node_dtype,count=n,offset=off)
assert n==meta['neuron_count'] and e==meta['edge_count']

W=sp.csr_matrix((edges['w'].astype(np.float32),(edges['pre'],edges['post'])),shape=(n,n),dtype=np.float32)
W.sum_duplicates()

# Full-connectome LIF smoke run. Sensory-region neurons are externally driven;
# all 139k-ish neurons remain in the recurrent state and can receive spikes.
rng=np.random.default_rng(20260929)
sens=np.flatnonzero(nodes['region']==0)
if len(sens)>512:
    sens=rng.choice(sens,512,replace=False)
v=np.full(n,-52.0,np.float32); g=np.zeros(n,np.float32); refr=np.zeros(n,np.int32)
rest=-52.; reset=-52.; th=-45.; dt=.1; tau_m=20.; tau_s=5.; wscale=.275
md=np.float32(np.exp(-dt/tau_m)); sd=np.float32(np.exp(-dt/tau_s)); gain=np.float32(1-md)
steps=300 # 30 ms biological time
history=[]; total_spikes=0; propagated_spikes=0
prev=np.empty(0,np.int64)
for step in range(steps):
    if prev.size:
        # outgoing full-connectome propagation from every presynaptic spike
        g += np.asarray(W[prev].sum(axis=0)).ravel().astype(np.float32)*wscale
    active=refr<=step
    v=np.where(active,rest+(v-rest)*md+g*gain,v); g*=sd
    fired=np.flatnonzero((v>th)&active)
    # Poisson external sensory drive around 120 Hz
    ext=sens[rng.random(len(sens))<(120*dt/1000)]
    spk=np.unique(np.concatenate([fired,ext])).astype(np.int64)
    if spk.size:
        v[spk]=reset; g[spk]=0; refr[spk]=step+22
    total_spikes+=int(spk.size); propagated_spikes+=int(fired.size)
    if step%30==0 or step==steps-1:
        history.append({'step':step,'time_ms':step*dt,'spikes_this_step':int(spk.size),'internally_generated':int(fired.size),'mean_v':float(v.mean())})
    prev=spk

report={
    'source':'snedea/flybrain data/connectome.bin.gz (FlyWire-derived packaged connectome)',
    'neurons':int(n),'edges_header':int(e),'csr_nonzero_edges':int(W.nnz),
    'sensory_region_neurons':int((nodes['region']==0).sum()),
    'stimulated_sensory_subset':int(len(sens)),
    'simulation_ms':steps*dt,'total_spikes':int(total_spikes),'internally_generated_spikes':int(propagated_spikes),
    'weight_min':float(edges['w'].min()),'weight_max':float(edges['w'].max()),'weight_mean':float(edges['w'].mean()),
    'history':history,
    'claim_boundary':'This runs LIF dynamics over the complete packaged connectome state. It is a computational model of FlyWire anatomy, not a biological consciousness simulation.'
}
(OUT/'full_connectome_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

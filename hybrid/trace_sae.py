import sys, os, torch
sys.path.insert(0, '/workspace/thinking-llms-interp')
sys.path.insert(0, '/workspace/thinking-llms-interp/hybrid')
from utils.sae import load_sae

think_id = 'open-reasoner-zero-7b'
sae, _ = load_sae(think_id, 20, 10, require_activation_mean=False)
print('After load_sae, sae.activation_mean norm:', float(sae.activation_mean.norm()))
print('  has _buffers["activation_mean"]?:', 'activation_mean' in sae._buffers)

ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'train-saes', 'results', 'vars', 'saes',
                          f'sae_{think_id}_layer20_clusters10.pt')
print('ckpt exists:', os.path.exists(ckpt_path))
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print('  ckpt keys:', list(ckpt.keys())[:10])
    if 'activation_mean' in ckpt:
        am = ckpt['activation_mean']
        print('  ckpt activation_mean norm:', float(am.norm()))
        sae.activation_mean = am
        print('  Loaded activation_mean from checkpoint')
    del ckpt
print('After ckpt overwrite, sae.activation_mean norm:', float(sae.activation_mean.norm()))
print('  is in _buffers now?:', 'activation_mean' in sae._buffers)
print('  first 5:', sae.activation_mean.tolist()[:5])

sae = sae.to('cpu')
print('After .to(cpu), norm:', float(sae.activation_mean.norm()))

act_mean = sae.activation_mean.detach().clone()
print('act_mean norm:', float(act_mean.norm()))

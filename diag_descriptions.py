import sys
sys.path.insert(0, '/workspace/thinking-llms-interp')
from utils.clustering import get_latent_descriptions
desc = get_latent_descriptions('open-reasoner-zero-7b', 20, 10)
print('type:', type(desc))
if isinstance(desc, dict):
    for k, v in desc.items():
        print(f"{k}: {str(v)[:120]!r}")
else:
    for d in desc:
        print(d)

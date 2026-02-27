import numpy as np
from multimodal_pso import pso_maximize, multimodal_15_12

n=2
bounds=[(-2*np.pi,2*np.pi)]*n
best_pos,best_val,history=pso_maximize(multimodal_15_12,bounds,n_dim=n,swarm_size=50,iters=200,seed=123,track_history=True)
Xhist=history['X_history']
final=Xhist[-1]
# find unique positions and evaluate value
uni=np.unique(final.round(6),axis=0)
print('unique positions:')
for pos in uni:
    val=multimodal_15_12(pos)
    print(pos, val)
print('best_val returned:', best_val, 'best_pos:', best_pos)

import numpy as np
X = np.load('data/processed/training_matrix/X.npy')
fold = np.load('data/processed/training_matrix/fold_ids.npy')
K = 411

phi = {}
for f, name in [(0,'S1'),(1,'S2'),(2,'S3'),(3,'P2')]:
    mask = fold == f
    phi[name] = set(np.where(X[mask][0, :K] != 0)[0])

print('Basis overlap:')
for a in ['S1','S2','S3','P2']:
    for b in ['S1','S2','S3','P2']:
        if a < b:
            print(f'  {a} and {b}: {len(phi[a] & phi[b])} shared bases')
import torch
import data
torch.set_printoptions(precision=3)

dims = (10, 10, 100)
balls = [(3, 3, 2, 0.3), 
         (2, 8, 1, 0.5),
         (7, 7, 2, 0.9)]

frames, R = data.gen_frames(dims, balls)
Dy, Dx, T = dims
K = len(balls)

# One Frame only
# R = R[:, :, 0, :].reshape(-1, 2)
R = R.reshape(Dx*Dy, T*2)
H = torch.rand((Dx*Dy, K))
W = torch.rand((K, T*2))

print('squared reconstruction error:', torch.norm(torch.matmul(H, W) - R))



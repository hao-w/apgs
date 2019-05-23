import torch
import data
import als
import models
torch.set_printoptions(precision=3)

dims = (100, 100, 100)
balls = [(3, 3, 2, 0.3), 
         (2, 8, 1, 0.5),
         (7, 7, 2, 0.9)]

frames, R = data.gen_frames(dims, balls)
Dy, Dx, T = dims
K = len(balls)

# One Frame only
# R = R[:, :, 0, :].reshape(-1, 2)
R = R.reshape(Dx*Dy, T*2)
H0 = torch.randn((Dx*Dy, K))
W0 = torch.randn((K, T*2))

H, W = als.solve_als(R, H0, W0)
print('squared reconstruction error:', torch.norm(torch.matmul(H, W) - R))

# def train(R, H0, W0):
#     H = H0.clone()
#     W = W0.clone()
    
#     fH = models.Enc_H(K)
#     fW = models.Enc_W(K)
#     optimizerH = torch.optim.Adam(list(fH.parameters()), lr=1e-2)
#     mse_lossH = torch.nn.MSELoss()
#     optimizerW = torch.optim.Adam(list(fW.parameters()), lr=1e-2)
#     mse_lossW = torch.nn.MSELoss()
#     regH = .1
#     regW = .1

#     for i in range(100):
#         for i in range(500):
#             WR = torch.stack([torch.matmul(W.detach(), Rt) for Rt in R])
#             H = fH(WR)
#             lossH = mse_lossH(torch.matmul(H, W.detach()), R) + regH*torch.norm(H, p=2)
#             # for param in fH.parameters():
#             #     lossH += regH * torch.norm(param, 2)
#             lossH.backward()
#             optimizerH.step()
#             optimizerH.zero_grad()
#         print('error:', torch.norm(torch.matmul(H, W) - R))

#         for i in range(500):
#             HR = torch.stack([torch.matmul(H.t().detach(), Rt) for Rt in R.t()]) 
#             W = fW(HR).t()
#             lossW = mse_lossW(torch.matmul(H.detach(), W), R) + regW*torch.norm(W, p=2)
#             # for param in fW.parameters():
#             #     lossH += regW * torch.norm(param, 2)
#             lossW.backward()
#             optimizerW.step()
#             optimizerW.zero_grad()

#         print('error2:', torch.norm(torch.matmul(H, W) - R))
#     return H, W

# H, W = train(R, H0, W0)

# def train_jointly(R, H0, W0):
#     H = H0.clone()
#     W = W0.clone()
    
#     fH = models.Enc_H(K)
#     fW = models.Enc_W(K)
#     optimizer = torch.optim.Adam(list(fW.parameters()) + list(fH.parameters()), lr=1e-4)
#     mse_loss = torch.nn.MSELoss()

#     # H = torch.zeros((Dx*Dy, K))
#     # W = torch.zeros((K, T*2))
#     for i in range(10):
#         optimizer.zero_grad()
        
#         H = fH(torch.stack([torch.matmul(W, Rt) for Rt in R]))
#         W = fW(torch.stack([torch.matmul(H.t(), Rt) for Rt in R.t()])).t()

#         loss = mse_loss(torch.matmul(H, W), R)
#         loss.backward()
#         optimizer.step()
#         print('error:', torch.norm(torch.matmul(H, W) - R))
#     return H, W

# H, W = train_jointly(R, H0, W0)

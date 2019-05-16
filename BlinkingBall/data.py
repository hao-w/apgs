import torch

def gen_frames(dims, balls):
    Dy, Dx, T = dims
    K = len(balls)
    
    frames = torch.zeros(dims)
    R = torch.zeros((*dims, 2))
    
    for bi, ball in enumerate(balls):
        bx, by, br, bp = ball
       
        for t in range(T):
            ba = Bernoulli(bp).sample()
            for y in range(by-br, by+br+1):
                for x in range(bx-br, bx+br+1):
                    if ((x-bx)**2 + (y-by)**2 <= br**2) and (x < Dx) and (y < Dy):
                        frames[y, x, t] = ba
                        R[y, x, t, int(ba.item())] = 1.
    return frames, R

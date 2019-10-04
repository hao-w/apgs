import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import probtorch
from utils import, Compose_IW

    def __init__(self, K, D, T, S, enc_coor, dec_coor, dec_digit, AT, mnist_mean):
        super().__init__()
        self.K = K
        self.D = D
        self.T = T
        self.S = S
        self.enc_coor = enc_coor
        self.dec_coor = dec_coor
        self.dec_digit = dec_digit
        self.AT = AT
        self.mnist_mean = mnist_mean

    def Update_where(frames, z_what=None, training=True):
        if z_what is None: ## meaning step 0
            digit = self.mnist_mean
        else:
            digit = self.dec_digit(z_what)
        for t in range(self.T):
            frame_t = frames[:,:,t, :,:]
            Qs = Sample_where_t(timestep=t, frame_t=frame_t, digit=self.mnist_mean)
            ## evaluate log_p and compute importance weights and losses
            ## resampleing
        return

    def Sample_where_t(self, frame_t, digit, training=True):
        S, B, K, _, _ = digit.shape
        frame_left = frame_t
        for k in range(K):
            digit_k = digit[:,:,k,:,:]
            conved_k = F.conv2d(frame_left.view(S*B, 64, 64).unsqueeze(0), digit_k.view(S*B, 28, 28).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ## T * S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            q = self.enc_coor.forward_0(conved_k)
            recon_frame_t_k = self.AT.digit_to_frame(digit_k.unsqueeze(2), q['z_where'].value.unsqueeze(2)).squeeze(2) ## S * B * 64 * 64
            frame_left = frame_left - recon_frame_t_k
            Qs.append(q)
        return Qs

========================
        q_where, log_p_where = os_coor(dec_coor, K=K, D=D, frames=frames, digit=mnist_mean)
        z_where = q_where['z_where'].value # S * B * T * K * D
        log_q_where = q_where['z_where'].log_prob.sum(-1).sum(-1).sum(-1) # S * B
        ##
        log_p_where = log_p_where.sum(-1).sum(-1).sum(-1)
        # print(log_p_where.shape)
        ##
        q_what, p_what = enc_digit(frames, z_where, crop)
        z_what = q_what['z_what'].value ## S * B * K * z_what_dim
        log_q_what = q_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        log_p_what = p_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        recon, ll = dec_digit(frames, z_what, crop, z_where=z_where)
        w = F.softmax(ll.sum(-1) + log_p_what + log_p_where - log_q_what - log_q_where, 0).detach()
        if training:
            phi_loss = (w * (- log_q_what - log_q_where)).sum(0).mean()
            theta_loss = (w * (- ll.sum(-1) - log_p_where)).sum(0).mean()
            ess = (1. / (w ** 2).sum(0)).mean()
            return phi_loss, theta_loss, ess, w, z_where, z_what
        else:
            ess = (1. / (w ** 2).sum(0)).mean()
            E_what =  q_what['z_what'].dist.loc.mean(0).detach()
            E_where = q_where['z_where'].dist.loc.mean(0).detach()
            return E_where, E_what, recon.detach(), ess, w, z_where, z_what, (w * (ll.sum(-1))).sum(0).mean().detach()

def Update_where(enc_coor, dec_coor, dec_digit, frames, crop, z_what, z_where_old, training=True):
    """
    update the z_where given the frames and digit images
    z_what  S * B * H* W
    """
    digit = dec_digit(frames, z_what, crop, z_where=None, intermediate=True)
    q_f_where, log_p_f = enc_coor(dec_coor, frames=frames, digit=digit)
    z_where = q_f_where['z_where'].value
    log_q_f = q_f_where['z_where'].log_prob.sum(-1).sum(-1).sum(-1) # S * B
    ##
    log_p_f = log_p_f.sum(-1).sum(-1).sum(-1)

    ##
    recon, ll_f = dec_digit(frames, z_what, crop, z_where=z_where)
    log_w_f = ll_f.sum(-1) + log_p_f - log_q_f

    q_b_where, log_p_b = enc_coor(dec_coor, frames=frames, digit=digit, sampled=False, z_where_old=z_where_old)
    _, ll_b = dec_digit(frames, z_what, crop, z_where=z_where_old)
    log_p_b = log_p_b.sum(-1).sum(-1).sum(-1)
    log_w_b = ll_b.sum(-1).detach() + log_p_b.detach() - q_b_where['z_where'].log_prob.sum(-1).sum(-1).sum(-1).detach()
    # log_w_b = ll_b.sum(-1).detach() + p_b_where['z_where'].log_prob.sum(-1).sum(-1).detach() - q_b_where['z_where'].log_prob.sum(-1).sum(-1).detach()
    if training:
        phi_loss, theta_loss, w = Compose_IW(log_w_f, log_q_f, log_w_b, ll_f.sum(-1)+ log_p_f)
        ess = (1. / (w**2).sum(0)).mean()
        return phi_loss, theta_loss, ess, w, z_where
    else:
        w = F.softmax(log_w_f - log_w_b, 0).detach()
        ess = (1. / (w**2).sum(0)).mean()
        E_where = q_f_where['z_where'].dist.loc.mean(0).detach()
        return E_where, recon.detach(), ess, w, z_where, (w * (ll_f.sum(-1))).sum(0).mean().detach()


def Resample_where(z_where, weights):
    S, B, K, dim4 = z_where.shape
    ancesters = Categorical(weights.transpose(0, 1)).sample((S, )).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, K, dim4) ## S * B * T * K * 2
    return torch.gather(z_where, 0, ancesters)

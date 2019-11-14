import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from  resampling import resample
"""
Amortized Population Gibbs objective in Bouncing MNIST problem
==========
abbreviations:
K -- number of digits
T -- timesteps in one bmnist sequence
S -- sample size
B -- batch size
ZD -- z_what_dim (ZD=10 in the paper)
FP -- square root of frame pixels (FP=96 in the paper)
DP -- square root of mnist digit pixels (DP=28 by default)
AT -- affine transformer
==========
variables:
frames : S * B * T * FP * FP, sequences of frames in bmnist, as data points
frame_t : S * B * FP * FP, frame at timestep t
z_where : S * B * T * K * 2, latent representaions of the trajectory, as local variables
z_what : S * B * K * ZD, latent representaions of the digits, as global variables
digit :  S * B * K * DP * DP, mnist digit templates used in convolution
mnist_mean : DP * DP,  mean of all the mnist images
===========
conv2d usage https://pytorch.org/docs/1.3.0/nn.functional.html?highlight=conv2d#torch.nn.functional.conv2d
    images: 1 * (SB) * FP * FP, kernels: (SB) * 1 * DP * DP, groups=(SB)
    ===> convoved: 1 * (SB) * (FP-DP+1) * (FP-DP+1)
===========
"""
    def __init__(self, models, AT, K, T, mnist_mean, frame_size, training=True):
        super().__init__()
        self.models = models
        (self.enc_coor, self.dec_coor, self.enc_digit, self.dec_digit) = self.models
        self.AT= AT
        self.K = K
        self.T = T
        self.mnist_mean = mnist_mean
        self.training = training
        self.frame_size = frame_size

    def Sweeps(self, apg_steps, S, B, frames):
        """
        Start with the mnist_mean template,
        and iterate over z_where_t and z_what
        """
        if self.training:
            metrics = {'phi_loss' : [], 'theta_loss' : [], 'ess' : [], 'log_joint' : []}
            phi_loss, theta_loss, trace_0 = self.Step0(S=S, B=B, frames=frames, training=self.training)
            z_where = trace_0['z_where']
            z_what = trace_0['z_what']
            metrics['phi_loss'].append(trace_0['phi_loss'].unsqueeze(0))
            metrics['theta_loss'].append(trace_0['theta_loss'].unsqueeze(0))
            metrics['ess'].append(trace_0['ess'].unsqueeze(0))
            metrics['log_joint'].append(trace_0['log_joint'].mean(0).unsqueeze(0))
            for m in range(apg_steps):
                trace_where = self.APG_where(S=S, B=B, frames=frames, z_what=z_what, z_where_old=z_where, training=self.training)
                z_where = trace_where['z_where']
                trace_what = self.APG_what(S=S, B=B, frames=frames, z_where=z_where, z_what_old=z_what, training=self.training)
                z_what = trace_what['z_what']
                metrics['phi_loss'].append((trace_what['phi_loss_what'] + trace_where['phi_loss_where']).unsqueeze(0))
                metrics['theta_loss'].append((trace_what['theta_loss_what'] + trace_where['theta_loss_where']).unsqueeze(0))
                metrics['ess'].append((trace_where['ess'] + trace_what['ess']).unsqueeze(0) / 2)
                metrics['log_joint'].append((trace_where['log_prior'] + trace_what['log_joint_p']).mean(0).unsqueeze(0))
            metrics['phi_loss'] = torch.cat(metrics['phi_loss'], 0)
            metrics['theta_loss'] = torch.cat(metrics['theta_loss'], 0)
            metrics['ess'] = torch.cat(metrics['ess'] , 0)
            metrics['log_joint'] = torch.cat(metrics['log_joint'], 0).mean(-1)
            return metrics
        else:
            metrics = {'recon' : [], 'E_where' : []}
            trace_0 = self.Step0(S=S, B=B, frames=frames, training=self.training)
            z_where = trace_0['z_where']
            z_what = trace_0['z_what']
            metrics['recon'].append(trace_0['recon'].unsqueeze(0))
            metrics['E_where'].append(trace_0['E_where'].unsqueeze(0))
            for m in range(apg_steps):
                trace_where = self.APG_where(S=S, B=B, frames=frames, z_what=z_what, z_where_old=z_where, training=self.training)
                z_where = trace_where['z_where']
                trace_what = self.APG_what(S=S, B=B, frames=frames, z_where=z_where, z_what_old=z_what, training=self.training)
                z_what = trace_what['z_what']
                metrics['recon'].append(trace_what['recon'].unsqueeze(0))
                metrics['E_where'].append(trace_where['E_where'].unsqueeze(0))
            metrics['E_where'] = torch.cat(metrics['E_where'], 0)
            metrics['recon'] = torch.cat(metrics['recon'], 0)
            return metrics

    def Step0(self, S, B, frames, training=True):
        mnist_mean_exp = self.mnist_mean.repeat(S, B, self.K, 1, 1)
        trace = {'z_where' : [], 'E_where' : [], 'z_what' : [], 'ess' : [], 'log_joint' : [], 'recon' : []}
        for t in range(self.T):
            frame_t = frames[:,:,t, :,:]

            if t == 0:
                trace_t = self.Where_1step(S=S, B=B, frame_t=frame_t, digit=mnist_mean_exp, z_where_t_1=None)
                log_p_where = trace_t['log_p']
                log_q_where = trace_t['log_q']
            else:
                trace_t = self.Where_1step(S=S, B=B, frame_t=frame_t, digit=mnist_mean_exp, z_where_t_1=trace_t['z_where_t'])
                log_p_where = log_p_where + trace_t['log_p']
                log_q_where = log_q_where + trace_t['log_q']

            trace['z_where'].append(trace_t['z_where_t'].unsqueeze(2)) ## S * B * 1 * K * D
            trace['E_where'].append(trace_t['E_where_t'].unsqueeze(1)) ## B * 1 * K * D
        trace['z_where'] = torch.cat(trace['z_where'], 2)
        trace['E_where'] = torch.cat(trace['E_where'], 1)
        cropped = self.AT.frame_to_digit(frames, trace['z_where']).view(S, B, self.T, self.K, 28*28)
        q_f_what, p_f_what = self.enc_digit(cropped)
        z_what_proposal = q_f_what['z_what'].value # S * B * K * z_what_dim
        log_q_what = q_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        log_p_what = p_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        recon, ll_f = self.dec_digit(frames, z_what_proposal, z_where=trace['z_where'])
        trace['log_joint'] = ll_f.sum(-1).detach().cpu() + log_p_what.cpu() + log_p_where.cpu()
        trace['recon'] = recon.detach().cpu()
        w = F.softmax(ll_f.sum(-1) + log_p_what - log_q_what + log_p_where - log_q_where, 0).detach()
        trace['ess'] = (1. / (w**2).sum(0)).mean().cpu().detach()
        trace['z_what'] = self.Resample_what(z_what_proposal, w)
        if training:
            phi_loss = (w * (- log_q_where - log_q_what)).sum(0).mean()
            theta_loss = (w * (-ll_f)).sum(0).mean()
            return phi_loss, theta_loss, trace
        else:
            return trace

    def Where_1step(self, S, B, frame_t, digit, z_where_t_1=None):
        frame_left = frame_t
        trace_t = {'log_p' : [], 'log_q' : [], 'z_where_t' : [], 'E_where_t' : []} ## a variable container
        for k in range(self.K):
            digit_k = digit[:,:,k,:,:]
            conved_k = F.conv2d(frame_left.view(S*B, self.frame_size, self.frame_size).unsqueeze(0), digit_k.view(S*B, 28, 28).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ##  S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            q_k = self.enc_coor.forward(conved_k)
            z_where_k = q_k['z_where'].value
            trace_t['z_where_t'].append(z_where_k.unsqueeze(2)) ## expand to S B 1 2
            trace_t['E_where_t'].append(q_k['z_where'].dist.loc.mean(0).cpu().unsqueeze(1))
            trace_t['log_q'].append(q_k['z_where'].log_prob.sum(-1).unsqueeze(-1)) # S * B * 1 --> K after loop

            if z_where_t_1 is not None:
                log_p_f_k = self.dec_coor.forward(z_where_k, z_where_t_1=z_where_t_1[:,:,k,:])
            else:
                log_p_f_k = self.dec_coor.forward(z_where_k)
            trace_t['log_p'].append(log_p_f_k.unsqueeze(-1))  # S * B * 1 --> K after loop
            recon_frame_t_k = self.AT.digit_to_frame(digit_k.unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2) ## S * B * 64 * 64
            frame_left = frame_left - recon_frame_t_k
        trace_t['log_p'] = torch.cat(trace_t['log_p'], -1).sum(-1) # S * B
        trace_t['log_q'] = torch.cat(trace_t['log_q'], -1).sum(-1) # S * B
        trace_t['z_where_t'] = torch.cat(trace_t['z_where_t'], 2) # S * B * K * D
        trace_t['E_where_t'] = torch.cat(trace_t['E_where_t'], 1) # B * K * D
        return trace_t

    def Where_apg_step(self, S, B, frame_t, digit, z_where_old_t, z_where_old_t_1=None, z_where_t_1=None):
        frame_left = frame_t
        trace_t = {'log_p_f' : [], 'log_p_b' : [],'log_q_f' : [], 'log_q_b' : [], 'z_where_t' : [], 'E_where_t' : []}
        for k in range(self.K):
            digit_k = digit[:,:,k,:,:]
            conved_k = F.conv2d(frame_left.view(S*B, self.frame_size, self.frame_size).unsqueeze(0), digit_k.view(S*B, 28, 28).unsqueeze(1), groups=int(S*B))
            CP = conved_k.shape[-1] # convolved output pixels ## T * S * B * CP * CP
            conved_k = F.softmax(conved_k.squeeze(0).view(S, B, CP, CP).view(S, B, CP*CP), -1) ## S * B * 1639
            q_k = self.enc_coor.forward(conved_k)
            z_where_k = q_k['z_where'].value
            trace_t['z_where_t'].append(z_where_k.unsqueeze(2)) ## expand to S B 1 2
            trace_t['E_where_t'].append(q_k['z_where'].dist.loc.mean(0).cpu().unsqueeze(1))
            trace_t['log_q_f'].append(q_k['z_where'].log_prob.sum(-1).unsqueeze(-1)) # S * B * 1 --> K after loop

            if z_where_t_1 is not None:
                log_p_f_k = self.dec_coor.forward(z_where_k, z_where_t_1=z_where_t_1[:,:,k,:])
            else:
                log_p_f_k = self.dec_coor.forward(z_where_k)
            ## backward
            log_q_b_k = Normal(q_k['z_where'].dist.loc, q_k['z_where'].dist.scale).log_prob(z_where_old_t[:,:,k,:]).sum(-1).detach()
            trace_t['log_q_b'].append(log_q_b_k.unsqueeze(-1))
            if z_where_old_t_1 is not None:
                log_p_b_k = self.dec_coor.forward(z_where_old_t[:,:,k,:], z_where_t_1=z_where_old_t_1[:,:,k,:])
            else:
                log_p_b_k = self.dec_coor.forward(z_where_old_t[:,:,k,:])

            trace_t['log_p_f'].append(log_p_f_k.unsqueeze(-1))
            trace_t['log_p_b'].append(log_p_b_k.unsqueeze(-1))  # S * B * 1 --> K after loop
            recon_frame_t_k = self.AT.digit_to_frame(digit_k.unsqueeze(2), z_where_k.unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2) ## S * B * 64 * 64
            frame_left = frame_left - recon_frame_t_k
        trace_t['log_p_f'] = torch.cat(trace_t['log_p_f'], -1).sum(-1) # S * B
        trace_t['log_p_b'] = torch.cat(trace_t['log_p_b'], -1).sum(-1) # S * B

        trace_t['log_q_f'] = torch.cat(trace_t['log_q_f'], -1).sum(-1) # S * B
        trace_t['log_q_b'] = torch.cat(trace_t['log_q_b'], -1).sum(-1) # S * B

        trace_t['z_where_t'] = torch.cat(trace_t['z_where_t'], 2) # S * B * K * D
        trace_t['E_where_t'] = torch.cat(trace_t['E_where_t'], 1) # B * K * D
        return trace_t

    def APG_where(self, S, B, frames, z_what, z_where_old, training=True):
        trace = {'w' : [], 'z_where' : [], 'E_where' : [], 'ess' : [], 'prior' : []}
        Phi_loss = []
        Theta_loss = []
        for t in range(self.T):
            frame_t = frames[:,:,t, :,:]
            digit = self.dec_digit(frame_t, z_what)
            if t == 0:
                trace_t = self.Where_apg_step(S=S, B=B, frame_t=frame_t, digit=digit, z_where_old_t=z_where_old[:,:,t, :, :], z_where_old_t_1=None, z_where_t_1=None)
                log_prior = trace_t['log_p_f']
            else:
                trace_t = self.Where_apg_step(S=S, B=B, frame_t=frame_t, digit=digit, z_where_old_t=z_where_old[:,:,t, :, :],  z_where_old_t_1=z_where_old[:,:,t-1, :,:], z_where_t_1=trace_t['z_where_t'])
                log_prior = log_prior + trace_t['log_p_f']
            log_w_f_t = trace_t['log_p_f'] - trace_t['log_q_f']
            log_w_b_t = trace_t['log_p_b'] - trace_t['log_q_b']

            _, ll_f_t = self.dec_digit(frame_t.unsqueeze(2), z_what, z_where=trace_t['z_where_t'].unsqueeze(2))

            _, ll_b_t = self.dec_digit(frame_t.unsqueeze(2), z_what, z_where=z_where_old[:,:,t,:,:].unsqueeze(2))
            w = F.softmax(log_w_f_t - log_w_b_t + ll_f_t.squeeze(-1) - ll_b_t.squeeze(-1), 0).detach()
            trace['ess'].append((1. / (w**2).sum(0)).mean().cpu().unsqueeze(-1))
            z_where_t = self.Resample_where(trace_t['z_where_t'], w)
            trace['z_where'].append(z_where_t.unsqueeze(2))
            trace['E_where'].append(trace_t['E_where_t'].unsqueeze(1))
            if training:
                Phi_loss.append((w * (- trace_t['log_q_f'])).sum(0).mean().unsqueeze(-1))
                Theta_loss.append((w * (- ll_f_t)).sum(0).mean().unsqueeze(-1))

        trace['z_where'] = torch.cat(trace['z_where'], 2)
        trace['E_where'] = torch.cat(trace['E_where'], 1)
        trace['ess'] = torch.cat(trace['ess'], -1).mean(-1)
        trace['prior'] = log_prior.cpu()
        if training:
            return torch.cat(Phi_loss, -1).sum(-1), torch.cat(Theta_loss, -1).sum(-1), trace
        else:
            return trace

    def APG_what(self, S, B, frames, z_where, z_what_old=None, training=True):
        trace = {'z_what' : [], 'ess' : [], 'recon' : [], 'log_joint_p' : []}
        croppd = self.AT.frame_to_digit(frames, z_where).view(S, B, self.T, self.K, 28*28)
        q_f_what, p_f_what = self.enc_digit(croppd)
        z_what_proposal = q_f_what['z_what'].value # S * B * K * z_what_dim
        log_q_f = q_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        log_p_f = p_f_what['z_what'].log_prob.sum(-1).sum(-1) # S * B
        recon, ll_f = self.dec_digit(frames, z_what_proposal, z_where=z_where)
        trace['log_joint_p'] =  ll_f.cpu().detach() + log_p_f.cpu()
        trace['recon'] = recon.cpu().detach()
        ## backward
        q_b_what, p_b_what = self.enc_digit(croppd, sampled=False, z_what_old=z_what_old)
        log_p_b = p_b_what['z_what'].log_prob.sum(-1).sum(-1).detach()
        log_q_b  = q_b_what['z_what'].log_prob.sum(-1).sum(-1).detach()
        _, ll_b = self.dec_digit(frames, z_what_old, z_where=z_where)
        w = F.softmax(ll_f.sum(-1) + log_p_f - log_q_f - (ll_b.sum(-1).detach() + log_p_b - log_q_b), 0).detach()
        trace['ess'] = (1. / (w**2).sum(0)).mean().cpu()
        trace['z_what'] = self.Resample_what(z_what_proposal, w)
        if training:
            phi_loss = (w * (- log_q_f)).sum(0).mean()
            theta_loss =(w * (- ll_f.sum(-1))).sum(0).mean()
            return phi_loss, theta_loss, trace
        else:
            return trace

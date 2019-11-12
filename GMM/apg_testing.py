import torch
from utils import shuffler

"""
==========
evaluation functions for apg samplers
==========
"""
def sample_data_uniform(DATAs, data_ptr):
    """
    ==========
    randomly select one gmm dataset from each group,
    given the data pointer
    ==========
    """
    datas = []
    for DATA in DATAs:
        datas.append(DATA[data_ptr])
    return datas

def test_single(model, apg_objective, apg_sweeps, datas, sample_size, CUDA, DEVICE):
    """
    ==========
    run apg sampler for each of the selected datasets
    ==========

    """
    loss_required = True
    ess_required = True
    mode_required = True
    density_required = True
    kl_required = True

    metrics = []
    for data in datas:
        data = data.unsqueeze(0).repeat(sample_size, 1, 1, 1)
        if CUDA:
            ob = data.cuda().to(DEVICE)
        trace = apg_objective(model=model,
                              apg_sweeps=apg_sweeps,
                              ob=ob,
                              loss_required=loss_required,
                              ess_required=ess_required,
                              mode_required=mode_required, # no need to track modes during training, since modes are only for visualization purpose at test time
                              density_required=density_required,
                              kl_required=kl_required)
        metrics.append(trace)
    return metrics








    def Test_budget(self, objective, Data, data_ptr, mcmc_steps, sample_size):
        Metrics = {'samples' : [], 'data' : [], 'log_joint' : [], 'ess' : []}
        batch = self.Sample_data_uniform(Data, data_ptr)
        for data in batch:
            Metrics['data'].append(data)
            data = data.repeat(sample_size, 1, 1, 1)
            if self.CUDA:
                with torch.cuda.device(self.device):
                    data = data.cuda()
            metrics = objective(self.models, data, mcmc_steps)
            # kl_step = kl_train(models, ob, reused, EPS)
            # Metrics['ess_eta'].append(metrics['ess_eta'][0, -1].unsqueeze(0))
            Metrics['ess'].append(metrics['ess'][0, -1].unsqueeze(0))

            Metrics['log_joint'].append(metrics['log_joint'][0, -1].unsqueeze(0))
        # kl_step = kl_train(models, ob, reused, EPS)

        return Metrics
    #
    def Test_ALL(self, objective, Data, mcmc_steps, Test_Params):
        Metrics = {'kl_ex' : [], 'kl_in' : [], 'll' : [], 'ess' : []}
        (S, B, DEVICE) = Test_Params
        EPS = torch.FloatTensor([1e-15]).log() ## EPS for KL between categorial distributions
        EPS = EPS.cuda().to(DEVICE) ## EPS for KL between categorial distributions
        for g, data_g in enumerate(Data):
            print("group=%d" % (g+1))
            NUM_DATASETS = data_g.shape[0]
            NUM_BATCHES = int((NUM_DATASETS / B))
            for step in range(NUM_BATCHES):
                ob = data_g[step*B : (step+1)*B]
                ob = shuffler(ob).repeat(S, 1, 1, 1)
                ob = ob.cuda().to(DEVICE)
                metrics = objective(self.models, ob, mcmc_steps, EPS)
                Metrics['ess'].append(metrics['ess'])
                Metrics['ll'].append(metrics['ll'])
                Metrics['kl_ex'].append(metrics['kl_ex'])
                Metrics['kl_in'].append(metrics['kl_in'])

        return Metrics

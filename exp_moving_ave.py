######
params = list(enc.parameters())+list(dec.parameters())
if batch_idx == 0 and epoch == 0:
    for j, p in enumerate(params):
        p_grad_np =p.grad.cpu()
        E_grads.append(p_grad_np)
        E_grads_2.append(torch.mul(p_grad_np, p_grad_np))
        accu_var += torch.sum(E_grads_2[j] - torch.mul(E_grads[j], E_grads[j])).data[0]
    beta1_t = beta1
    beta2_t = beta2

else:
    E_grads_factor = beta1 *  (1 - beta1_t) / (1-beta1_t * beta1)
    gt_factor = (1 - beta1) / (1 - beta1_t * beta1)

    E_grads_2_factor = beta2 * (1 - beta2_t) / (1-beta2_t * beta2)
    gt_2_factor = (1 - beta2) / (1-beta2_t * beta2)

    for j, p in enumerate(params):
        p_grad_np = p.grad.cpu()
        E_grads[j] = E_grads_factor * E_grads[j]  + gt_factor * p_grad_np
        E_grads_2[j] = E_grads_2_factor * E_grads_2[j]  + gt_2_factor  *  torch.mul(p_grad_np, p_grad_np)
        a = torch.sum(E_grads_2[j] - torch.mul(E_grads[j], E_grads[j]))
        b = torch.sum((E_grads_2[j] + 1e-9)
                      /
                      (torch.mul(E_grads[j], E_grads[j]) + 1e-9)
                      - 1)
        accu_mean2 += b.data[0]
        accu_var += a.data[0]
    accu_mean2 /= N_params
    accu_var /= N_params
    beta1_t = beta1_t * beta1
    beta2_t = beta2_t * beta2

mean2s_list.append(accu_mean2)
vars_list.append(accu_var)
######

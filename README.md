# ACV-FL: Adaptive Communication Optimization for Collusion-Resistant Verification Federated Learning
ACV is a generic framework for federated learning for collusion-resistant verification with adaptive communication optimization (AOTop-k).

## Experimental Environment
* Ubuntu 20.04.5 LTS PC
* Python 3.10
* PyTorch 2.0.1
* NumPy 1.26.0

## Experimental setup
* learning rate = 0.01
* momentum = 0.5
* batch size = 64

## Main Code

``` python

num = result.numel()
ab = [0.5, 0.5]  # alpha and beta

self.result_list.append(result)
if len(self.result_list) > 2:
  self.result_list.pop(0)

k = self.k_prime(self.result_list, num, ab)  # AOTop-k Alogithm
result = self.AOTopk(self.result_list, k, ab)
print('size=',self.calcu_comm_cost(result))
utils.vec2net(result, Delta)

return Delta

def k_prime(self, result, num, ab):
  k_min = int(num / 100)
  k_avg = int(num / 10)
  k_max=int(num / 10)
  eucl_distance1 = torch.norm(result[1] - result[0])
  eucl_distance2 = torch.norm(result[1])
  k_kk = (ab[0] * eucl_distance2) / (ab[1] * eucl_distance1)
  k = int(max(k_min, min(k_max, k_min * k_kk)))
  format_k = "{:.3f}".format(k / num)
  self.k_list.append(format_k)
  print('k=', self.k_list)
  return k

def AOTop_k(self, result, k,ab):
  alpha_1 = 0.5
  beta_1 = 0.5
  Ma = ab[0] * torch.abs(result[1])
  Va = ab[1] * torch.abs(result[1] - result[0])
  S = Ma + Va

  topk_value, topk_ind = torch.topk(S, k)
  result_k = torch.zeros_like(result[1])
  result_k[topk_ind] = result[1][topk_ind]

  result_dd = result[1].clone()
  result_dd[topk_ind] = 0
  self.result_add = result_dd + self.result_add

  return result_k
```

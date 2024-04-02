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

## Datasets
* MNIST
* CIFAR-10
* CIFAR-100

## Data distribution
* IID
* Non-IID

## Parameters
* Number of clients
* Average $p$ value
* gradient magnitude $\alpha$ and gradient variation $\beta$

## Main Code

``` python
def Main(self, clients, func)：
  ***
  ***
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

def calcu_comm_cost(self,result):
  gradient_byte = result.data.numpy().tobytes()
  gradient_size = len(gradient_byte)
  mb = gradient_size/(1024.0*1024.0)
  return mb

def k_prime(self, result, num, ab):
  k_min = int(num / 100)
  k_opt = int(num / 10)
  k_max=int(num / 10)
  eucl_distance1 = torch.norm(result[1] - result[0])
  eucl_distance2 = torch.norm(result[1])
  k_kk = (ab[0] * eucl_distance2) / (ab[1] * eucl_distance1)
  k = int(max(k_min, min(k_max, k_opt * k_kk)))
  format_k = "{:.3f}".format(k / num)
  self.k_list.append(format_k)
  print('k=', self.k_list)
  return k

def AOTop_k(self, result, k,ab):
  Ma = ab[0] * torch.abs(result[1])
  Va = ab[1] * torch.abs(result[1] - result[0])
  S = Ma + Va

  topk_value, topk_ind = torch.topk(S, k)
  result_k = torch.zeros_like(result[1])
  result_k[topk_ind] = result[1][topk_ind]

  result_dd = result[1].clone()
  result_dd[topk_ind] = 0
  self.result_add = result_dd + self.result_add
  self.i = self.i + k
  if self.i > 81194:  #gradient dimension
      topkdd_value, topkdd_ind = torch.topk(torch.abs(self.result_add), int(k/2))
      resultdd_k = torch.zeros_like(self.result_add)
      resultdd_k[topkdd_ind] = self.result_add[topkdd_ind]
      result_k = result_k +resultdd_k
      self.result_add[topkdd_ind] = 0
      self.i = 0

  return result_k
```

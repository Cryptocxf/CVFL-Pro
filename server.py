from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn.functional as F

from utils import utils
from utils.backdoor_semantic_utils import SemanticBackdoor_Utils
from utils.backdoor_utils import Backdoor_Utils
import time


class Server():
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cpu'):
        # nll_loss负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
        self.clients = []
        self.k_list = []
        self.result_list = [torch.zeros(81194)]
        self.result_add = torch.zeros(81194)
        self.model = model
        self.dataLoader = dataLoader  # 数据加载
        self.device = device  # 设备
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.i = 0
        self.AR = self.FedAvg
        self.func = torch.mean
        self.isSaveChanges = False
        self.savePath = './AggData'
        self.criterion = criterion
        self.path_to_aggNet = ""

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    # 分发模型
    # 设定client的模型
    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        print("[Server] Start testing")
        self.model.to(self.device)  # GPU
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                if output.dim() == 1:
                    pred = torch.round(torch.sigmoid(output))
                else:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, count,
                                                                                          accuracy))
        return test_loss, accuracy

    def test_backdoor(self):
        print("[Server] Start testing backdoor\n")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = Backdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set (Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                                           correct,
                                                                                                           len(
                                                                                                               self.dataLoader.dataset),
                                                                                                           accuracy))
        return test_loss, accuracy

    def test_semanticBackdoor(self):
        print("[Server] Start testing semantic backdoor")

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        utils = SemanticBackdoor_Utils()
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = utils.get_poison_batch(data, target, backdoor_fraction=1,
                                                      backdoor_label=utils.backdoor_label, evaluation=True)
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                # 值最大的那个即对应着分类结果，然后把分类结果保存在 pred 里
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.dataLoader.dataset)
        accuracy = 100. * correct / len(self.dataLoader.dataset)

        self.model.cpu()  ## avoid occupying gpu when idle
        print(
            '[Server] Test set (Semantic Backdoored): Average loss: {:.4f}, Success rate: {}/{} ({:.0f}%)\n'.format(
                test_loss,
                correct,
                len(
                    self.dataLoader.dataset),
                accuracy))
        return test_loss, accuracy, data, pred

    # 训练
    # 从服务端控制和模拟client的训练，参数为训练的client集合
    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()

        if self.isSaveChanges:
            self.saveChanges(selectedClients)

        tic = time.perf_counter()
        Delta = self.AR(selectedClients)
        self.Delta = Delta
        toc = time.perf_counter()
        print(f"[Server] The aggregation takes {toc - tic:0.6f} seconds.\n")

        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def saveChanges(self, clients):

        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]

        param_trainable = utils.getTrainableParameters(self.model)

        param_nontrainable = [param for param in Delta.keys() if param not in param_trainable]
        for param in param_nontrainable:
            del Delta[param]
        print(f"[Server] Saving the model weight of the trainable paramters:\n {Delta.keys()}")
        for param in param_trainable:
            ##stacking the weight in the innerest dimension
            param_stack = torch.stack([delta[param] for delta in deltas], -1)
            shaped = param_stack.view(-1, len(clients))
            Delta[param] = shaped

        saveAsPCA = False
        saveOriginal = True
        if saveAsPCA:
            from utils import convert_pca
            proj_vec = convert_pca._convertWithPCA(Delta)
            savepath = f'{self.savePath}/pca_{self.iter}.pt'
            torch.save(proj_vec, savepath)
            print(
                f'[Server] The PCA projections of the update vectors have been saved to {savepath} (with shape {proj_vec.shape})')
        #             return
        if saveOriginal:
            savepath = f'{self.savePath}/{self.iter}.pt'

            torch.save(Delta, savepath)
            print(f'[Server] Update vectors have been saved to {savepath}')

    ## Aggregation functions ##
    # 设置聚合方案
    def set_AR(self, ar):
        if ar == 'fedavg':
            self.AR = self.FedAvg
        elif ar == 'krum':
            self.AR = self.krum
        elif ar == 'mkrum':
            self.AR = self.mkrum
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")

    # 联邦平均
    def FedAvg(self, clients):
        # torch.mean求平均
        # 被选中在客户端数量（5/10个）
        # print('clients=',clients)
        # dim指维度，输入为(m,n,k)，dim=-1，则输出(m,n,1)或(m,n)，keepdim选择是否需要保持结构，即1
        out = self.FedFuncWholeNet(clients, lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        # print('out=',out)
        return out

    # krum
    def krum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('krum').cpu()(arr.cpu()))
        return out

    # mkrum
    def mkrum(self, clients):
        from rules.multiKrum import Net
        self.Net = Net
        out = self.FedFuncWholeNet(clients, lambda arr: Net('mkrum').cpu()(arr.cpu()))
        return out

    def FedFuncWholeNet(self, clients, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        # 将更新向量视为堆叠向量（单行，d维，n个）
        # deepcopy()可以复制的列表里包含子列表，但copy()不可以
        Delta = deepcopy(self.emptyStates)
        # print('Delta1=', Delta)
        # 获得所有客户端的模型更新
        deltas = [c.getDelta() for c in clients]
        # print('delta=',deltas)
        # print('end*******************')
        # 模型更新转化为一维向量
        vecs = [utils.net2vec(delta) for delta in deltas]
        print('vecs1=', len(vecs[0]), len(vecs))
        # isfinite返回一个带有布尔元素的新张量，表示每个元素是否是有限的。
        # 当实数值不是 NaN、负无穷或无穷大时，它们是有限的。当复数值的实部和虚部都是有限的时，复数值是有限的。
        # tensor.all 如果张量tensor中所有元素都是True, 才返回True; 否则返回False
        # tensor.item 该方法的功能是以标准的Python数字的形式来返回这个张量的值。这个方法只能用于只包含一个元素的张量。
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        # print('vecs2=', vecs)
        # stack() 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        # unsqueeze在指定的位置插入一个维度，有两个参数，input是输入的tensor,dim是要插到的维度
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        # print('result1=', result)
        # torch.view(x, -1) & torch.view(-1)将原 tensor 以参数 x 设置第一维度重排，第二维度自动补齐；当没有参数 x 时，直接重排为一维的 tensor
        result = result.view(-1)
        print('size=', self.calcu_comm_cost(result))
        num = result.numel()
        ab = [0.5, 0.5]  # alpha和beta对应在值
        # print('result=', result)
        # k = int(num/10)
        # print('list=', self.result_list)
        self.result_list.append(result)
        if len(self.result_list) > 2:
            self.result_list.pop(0)
        # print('list=',self.result_list)
        k = self.k_prime(self.result_list, num, ab)  # 自适应Top-k选择算法
        # print('result_add=',self.result_add)
        # k=int(num/20)
        # Ma = torch.abs(result)
        # topk_value, topk_ind = torch.topk(Ma, k)
        # result_k = torch.zeros_like(result)
        # result_k[topk_ind] = result[topk_ind]
        # result = result_k
        result = self.AOTopk(self.result_list, k, ab)
        print('size=',self.calcu_comm_cost(result))
        # print('result2=', result,self.result_add)
        utils.vec2net(result, Delta)
        # print('Delta2=', Delta)
        return Delta

    def calcu_comm_cost(self,result):
        gradient_bytes = result.data.numpy().tobytes()
        gradient_size = len(gradient_bytes)
        mb = gradient_size/(1024.0*1024.0)
        return mb

    def k_prime(self, result, num, ab):
        k_min = int(num / 100)
        k_opt = int(num / 10)
        k_max=int(num / 10)
        eucl_distance1 = torch.norm(result[1] - result[0])
        eucl_distance2 = torch.norm(result[1])
        k_kk = (ab[0] * eucl_distance2) / (ab[1] * eucl_distance1)
        print('kkk=',k_kk,k_avg* k_kk)
        k = int(max(k_min, min(k_max, k_opt * k_kk)))
        format_k = "{:.3f}".format(k / num)
        self.k_list.append(format_k)
        print('k=', self.k_list)
        return k

    def AOTopk(self, result, k, ab):

        Ma = ab[0] * torch.abs(result[1])
        Va = ab[1] * torch.abs(result[1] - result[0])
        S = Ma + Va
        # print('S=',S)
        topk_value, topk_ind = torch.topk(S, k)
        # topk_value, topk_ind = torch.topk(Ma, k)
        # print('ind=',topk_ind)
        result_k = torch.zeros_like(result[1])
        result_k[topk_ind] = result[1][topk_ind]

        result_dd = result[1].clone()
        result_dd[topk_ind] = 0
        self.result_add = result_dd + self.result_add
        self.i = self.i + k
        print('i=', self.i)
        if self.i > 8119:
            topkdd_value, topkdd_ind = torch.topk(torch.abs(self.result_add), int(k / 2))
            resultdd_k = torch.zeros_like(self.result_add)
            resultdd_k[topkdd_ind] = self.result_add[topkdd_ind]
            result_k = result_k + resultdd_k
            self.result_add[topkdd_ind] = 0
            self.i = 0
        return result_k

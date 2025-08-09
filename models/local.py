import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from models.utils.federated_model import FederatedModel

class Local(FederatedModel):
    NAME = 'local'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(Local, self).__init__(nets_list, args, transform)

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

		# 进行本地训练
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
		
		# 更新global_net为训练后的某个参与者网络（用于评估展示）
		# 使用第一个在线客户端的网络作为代表
        if online_clients:
            self.global_net.load_state_dict(self.nets_list[online_clients[0]].state_dict())
		
        return None

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f (No Aggregation)" % (index, loss)
                optimizer.step()
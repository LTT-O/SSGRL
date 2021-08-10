import torch
from torch import nn
import torch.nn.functional as F


class SemanticDecoupling(nn.Module):
    def __init__(self, num_classes, image_feature_dim, word_feature_dim, intermediary_dim=1024):
        '''
        :param num_classes:     CoCo - 80
        :param image_feature_dim:   After ResNet101 2048
        :param word_feature_dim:    After Glove 300
        :param intermediary_dim:    d1,d2 - 1024
        '''
        super(SemanticDecoupling, self).__init__()
        self.num_classes = num_classes  # 80
        self.image_feature_dim = image_feature_dim  # N
        self.word_feature_dim = word_feature_dim  # ds
        self.intermediary_dim = intermediary_dim  # d1,d2

        self.fc1 = nn.Linear(image_feature_dim, intermediary_dim, bias=False)  # U
        self.fc2 = nn.Linear(word_feature_dim, intermediary_dim, bias=False)  # V
        self.fc3 = nn.Linear(intermediary_dim, intermediary_dim, bias=True)  # P
        self.fc4 = nn.Linear(intermediary_dim, 1)  # f_a

    def forward(self, img_feature_map, word_features):
        '''
        :param img_feature_map: shape = batch_size * channel * img_size * img_size
        :param word_features:    shape = classNum * wordFeatureDim - 80 * 300
        :return:The input of GNN
        '''
        batch_size, img_size = img_feature_map.shape[0], img_feature_map.shape[3]
        img_feature_map = torch.transpose(torch.transpose(img_feature_map, 1, 2), 2,
                                          3)  # batch_size * img_size * img_size * channel
        img_feature = img_feature_map.contiguous().view(batch_size * img_size * img_size,
                                                        -1)  # (batch_size * img_size * img_size) * channel
        img_feature = self.fc1(img_feature).view(batch_size * img_size * img_size, 1, -1).repeat(1, self.num_classes,
                                                                                                 1)  # (batch_size * img_size * img_size) * classes * intermediary_dim

        word_feature = self.fc2(word_features).view(1, self.num_classes, self.intermediary_dim).repeat(
            batch_size * img_size * img_size, 1, 1)  # (batch_size * img_size * img_size) * classes * intermediary_dim

        feature = self.fc3(torch.tanh(img_feature * word_feature).view(-1,
                                                                       self.intermediary_dim))  # (batch_size * img_size * img_size * classes) * intermediary_dim

        coefficient = self.fc4(feature)  # (batch_size * img_size * img_size * classes) * 1

        coefficient = coefficient.view(batch_size, img_size, img_size,
                                       self.num_classes)  # batch_size * img_size * img_size * classes
        coefficient = torch.transpose(torch.transpose(coefficient, 2, 3), 1, 2).view(
            batch_size, self.num_classes, -1)  # batch_size * classes * (img_size * img_size)

        coefficient = F.softmax(coefficient, dim=2)  # batch_size * classes * (img_size * img_size)
        coefficient = coefficient.view(batch_size, self.num_classes, img_size,
                                       img_size)  # batch_size * classes * img_size * img_size
        coefficient = torch.transpose(torch.transpose(coefficient, 1, 2), 2,
                                      3)  # batch_size  * img_size * img_size * classes
        coefficient = coefficient.view(batch_size, img_size, img_size, self.num_classes, 1).repeat(
            1, 1, 1, 1, self.image_feature_dim)  # batch_size  * img_size * img_size * classes * image_feature_dim

        img_feature_map = img_feature_map.view(batch_size, img_size, img_size, 1, self.image_feature_dim).repeat(
            1, 1, 1, self.num_classes, 1)  # # batch_size  * img_size * img_size * classes * image_feature_dim

        graph_net_input = torch.sum(torch.sum(coefficient * img_feature_map, 1), 1)

        return graph_net_input

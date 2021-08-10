import torch
import numpy as np
from torch import nn
from model.SD import SemanticDecoupling
from model.GGNN import GatedGNN
from model.element_wise_layer import Element_Wise_Layer
from model.resnet import resnet101


class SSGRL(nn.Module):

    def __init__(self, imageFeatureDim, intermediaDim,
                 outputDim, adjacencyMatrixPath, wordFeaturesPath,
                 classNum=80, wordFeatureDim=300, timeStep=3):
        super(SSGRL, self).__init__()
        self.backbone = resnet101()
        self.timeStep = timeStep
        self.classNum = classNum
        self.imageFeatureDim = imageFeatureDim  # 2048 output of ResNet101
        self.intermediaDim = intermediaDim      # d1,d2-联合嵌入和输出特征的尺寸,SD模块中
        self.outputDim = outputDim              # 2048的输出，GGNN的输出为2048
        self.wordFeatureDim = wordFeatureDim    # X_c的维度，Glove模块出来

        self.wordFeatures = self.load_features(wordFeaturesPath)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrixPath)   # 入/出 度邻接矩阵

        self.SemanticDecoupling = SemanticDecoupling(self.classNum, self.imageFeatureDim,   # SD模块
                                                     self.wordFeatureDim, intermediary_dim=self.intermediaDim)
        self.GGNN = GatedGNN(self.imageFeatureDim, self.timeStep, self.inMatrix, self.outMatrix)    # GGNN模块

        self.fc = nn.Linear(2 * self.imageFeatureDim, self.outputDim)
        self.classifiers = Element_Wise_Layer(self.classNum, self.outputDim)    # 全连接层，输出概率值

    def forward(self, input):
        batch_size = input.shape[0]
        # ResNet-101
        featuremap = self.backbone(input)   # (BatchSize, Channel, imgSize, imgSize)

        # SD
        semanticFeature = self.SemanticDecoupling(featuremap, self.wordFeatures)    # (BatchSize, classNum, imageFeatureDim)

        # GGNN
        feature = self.GGNN(semanticFeature)    # (BatchSize, classNum, imageFeatureDim)
        feature = torch.cat((feature.view(batch_size * self.classNum, -1),
                             semanticFeature.view(-1, self.imageFeatureDim)), 1)  # (BatchSize, classNum, 2*imageFeatureDim)
        output = torch.tanh(self.fc(feature))
        output = output.contiguous().view(batch_size, self.classNum, self.outputDim)  # (BatchSize, classNum, outputDim)
        result = self.classifiers(output)  # (BatchSize, classNum)

        return result

    def load_features(self, wordFeaturesPath):
        return nn.Parameter(torch.from_numpy(np.load(wordFeaturesPath).astype(np.float32)), requires_grad=False)

    def load_matrix(self, adjacencyMatrixPath):
        mat = np.load(adjacencyMatrixPath)
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(
            torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix

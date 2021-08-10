import torch
import torch.nn as nn


class GatedGNN(nn.Module):

    def __init__(self, inputDim, timeStep, inMatrix, outMatrix):
        super(GatedGNN, self).__init__()

        self.inputDim, self.timeStep, self.inMatrix, self.outMatrix = inputDim, timeStep, inMatrix, outMatrix
        self.fc1_W, self.fc1_U = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)
        self.fc2_W, self.fc2_U = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)
        self.fc3_W, self.fc3_U = nn.Linear(2 * inputDim, inputDim), nn.Linear(inputDim, inputDim)

    def forward(self, input):
        """
        :param input:(BatchSize, classNum, inputDim)
               Shape of adjMatrix : (classNum, BatchSize, BatchSize)
        :return allNodes:（BatchSize, classNum, imgFeatureDim）
        """
        batch_size, nodeNum = input.shape[0], self.inMatrix.shape[0]
        allNodes = input
        inMatrix = self.inMatrix.repeat(batch_size, 1).view(batch_size, nodeNum, -1)
        outMatrix = self.outMatrix.repeat(batch_size, 1).view(batch_size, nodeNum, -1)

        for i in range(self.timeStep):
            a_c = torch.cat((torch.bmm(inMatrix, allNodes), torch.bmm(outMatrix, allNodes)), 2)  # BatchSize * nodeNum * (2 * inputDim)
            a_c = a_c.contiguous().view(batch_size*nodeNum, -1)  # (BatchSize * nodeNum) * (2 * inputDim)

            flatten_allNodes = allNodes.view(batch_size * nodeNum, -1)  # (BatchSize * nodeNum) * inputDim

            z_c = torch.sigmoid(self.fc1_W(a_c) + self.fc1_U(flatten_allNodes))  # (BatchSize * nodeNum) * inputDim

            r_c = torch.sigmoid(self.fc2_W(a_c) + self.fc2_U(flatten_allNodes))  # (BatchSize * nodeNum) * inputDim

            h_c = torch.tanh(self.fc3_W(a_c) + self.fc3_U(r_c * flatten_allNodes))  # (BatchSize * nodeNum) * inputDim

            flatten_allNodes = (1 - z_c) * flatten_allNodes + z_c * h_c  # (BatchSize * nodeNum) * inputDim

            allNodes = flatten_allNodes.view(batch_size, nodeNum, -1)  # BatchSize * nodeNum * inputDim

        return allNodes

import torch.nn as nn
import torch
import torch.nn.functional as func


# from pytorch official website

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        input = input * self.weight
        batch = self.target.size(0)
        size = self.target.size(1)
        target = self.target.view(batch*self.target.size(1), -1)
        input = input.view(batch*input.size(1), -1)

        temp = nn.functional.pairwise_distance(input,target)
        temp = temp.view(batch,size)
        loss = torch.mean(temp)
        return loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        IG = self.gram(input*self.weight)
        TG = self.gram(self.target)
        self.loss = self.criterion(IG, TG)
        return self.loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss

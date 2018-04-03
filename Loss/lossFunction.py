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
        a, b, c, d = input.size()
        input = input * self.weight
        target = self.target.view(a*b, -1)
        input = input.view(a*b, -1)

        temp = nn.functional.pairwise_distance(input,target)
        temp = temp.div(c*d)
        temp = temp.view(a,b)
        loss = torch.mean(temp)
        return loss
    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #     return self.loss


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        temp = torch.chunk(input, a, dim=0)
        result = []
        for i in temp:
            features = i.view(b, c * d)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            result.append(G.div(b * c * d))

        return torch.cat(tuple(result),dim=0)

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

from net.CDCN import *
from Loss.lossFunction import *
from net import VGG
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from data.dataloader import *
import torchvision.transforms as transforms

log = False

net = VGG.modify_vgg()
net.eval()
net_root = './netWeight/'


# this two parameters are whole image with the same size
def lossFunction(output, target, c_s, use_cuda,net):
    if c_s == 0:
        c_target = net.get_feature('ReLU', 5, 1, target)
        c_output = net.get_feature('ReLU', 5, 1, output)
        if use_cuda:
            c_loss = ContentLoss(c_target, 1).cuda()
        else:
            c_loss = ContentLoss(c_target, 1)
        loss = c_loss(c_output)
        return loss
    else:
        s_target = net.get_feature('ReLU', 3, 1, target)
        s_output = net.get_feature('ReLU', 3, 1, output)
        if use_cuda:
            s_loss = StyleLoss(s_target, 1).cuda()
        else:
            s_loss = StyleLoss(s_target, 1)
        loss = s_loss(s_output)
        return loss


def train(model, dataloader, optimizer, epoch, n_epochs, use_cuda, c_s,net):
    # the model of training
    if log:
        shower = transforms.ToPILImage()
    model.train()
    running_loss = 0.0
    print("-" * 10)
    print('Epoch {}/{}'.format(epoch, n_epochs))
    for data in dataloader:
        x_train, y_train = data
        #print(x_train.size())
        if log:
            shower(x_train.squeeze()).convert('RGB').show()
        if use_cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        # change data to Variable, a wrapper of Tensor
        x_train, y_train = Variable(x_train), Variable(y_train)

        optimizer.zero_grad()
        outputs = model(x_train)
        if log:
            shower(outputs.data.squeeze()).convert('RGB').show()

        loss = lossFunction(outputs, y_train, c_s, use_cuda,net)
        loss.backward()

        # optimize the weight of this net
        optimizer.step()
        running_loss += loss.data[0]

    print("Loss {}".format(running_loss / len(dataloader)))
    print("-" * 10)


def test(model, testloader, use_cuda, c_s,net):
    model.eval()
    test_loss = 0.0
    print("-" * 10)
    print("test process")

    for data in testloader:
        x_test, y_test = data
        if use_cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()
        x_test, y_test = Variable(x_test), Variable(y_test)
        output = model(x_test)

        test_loss += lossFunction(output, y_test, c_s, use_cuda,net).data[0]
    print("Loss {}".format(test_loss / len(testloader)))
    print("-" * 10)


# c_s: if the value is 0, use content loss function, or use style loss function
def init(net_type, c_s,net):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    model = CDCN()
    if not os.path.exists(net_root):
        os.mkdir(path=net_root)
    path = net_root + net_type
    if use_cuda:
        model.cuda()
        net.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
        cudnn.benchmark = True
        path += '_cuda'
    print('begin')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        optimizer = torch.optim.Adam(model.parameters())
        train_set = get_train_data()
        n_epochs = 200
        if log:
            n_epochs = 1

        for i in range(n_epochs):
            train(model=model,
                  dataloader=train_set,
                  optimizer=optimizer,
                  epoch=i,
                  n_epochs=n_epochs,
                  use_cuda=use_cuda,
                  c_s=c_s,
                  net=net)
        torch.save(model.state_dict(), path)
        print('successfully save weights')


def main():
    net = VGG.modify_vgg()
    net.eval()
    init("content", 0,net)
    init("style", 1,net)


if __name__ == '__main__':
    main()

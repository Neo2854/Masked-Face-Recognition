import numpy as np

from torch import nn
from torch import optim
from facenet_pytorch import InceptionResnetV1

from matplotlib import pyplot as plt

SGD = 0
ADA_GRAD = 1
ADAM = 2

class Facenet(nn.Module):
    def __init__(self, pretrained = False, num_classes = 100) -> None:
        super(Facenet, self).__init__()

        if pretrained:
            self.inception_net = InceptionResnetV1(pretrained="vggface2").eval()
            self.num_classes = 128

            for param in self.inception_net.parameters():
                param.requires_grad = False
        else:
            self.inception_net = InceptionResnetV1(num_classes=num_classes)
            self.num_classes = num_classes

        self.linear = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256, eps=0.01, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(256, num_classes, bias=True)
        )

        self.train_loss_v_epoch = None
        self.valid_loss_v_epoch = None
        self.test_loss_v_epoch = None

    def forward(self, x):
        x = self.inception_net(x)
        x = self.linear(x)

        return x

    def triplet_loss(self, anchor, positive, negative, alpha = 5):
        return ((anchor - positive)**2 - (anchor - negative)**2).sum() + alpha

    def fit(self, train_loader, validation_loader = None, epochs = 10, optimizer = ADAM, lr = 0.0001, optim_args = dict(), verbose = 1):
        # Function for training
        # train_loader and validation_loader are objects of DataLoader class from PyTorch
        # Depending on the optimizer used required optimizer args which are in PyTorch can be passed
            #   Example: if optmizer = SGD
            #   optim_args = {"momemtum" : 0.01}

        self.train_loss_v_epoch = np.zeros((2, epochs))
        self.valid_loss_v_epoch = np.zeros((2, epochs))

        if optimizer == SGD:
            pass
        elif optimizer == ADA_GRAD:
            pass
        else:
            optimizer = optim.Adam(self.parameters(), lr = lr)

        for epoch in range(epochs):
            print("For epoch {}".format(epoch+1))

            self.train(True)

            train_loss = 0
            count = 0
            for i, data in enumerate(train_loader):
                anchors, positives, negatives = data

                optimizer.zero_grad()
                anchors, positives, negatives = self.forward(anchors), self.forward(positives), self.forward(negatives)
                
                loss = self.triplet_loss(anchors, positives, negatives)

                loss.backward()
                optimizer.step()

                curr_loss = loss.item()
                train_loss += curr_loss
                count += 1

                print("\t\tFor Epoch {}, Loss for Batch {} = {}".format(epoch + 1, i + 1, curr_loss))

            avg_train_loss = train_loss/count

            self.train(False)

            if validation_loader:
                valid_loss = 0
                count = 0
                for i, data in enumerate(validation_loader):
                    anchors, positives, negatives = data

                    optimizer.zero_grad()
                    anchors, positives, negatives = self.forward(anchors), self.forward(positives), self.forward(negatives)
                
                    loss = self.triplet_loss(anchors, positives, negatives)
                    valid_loss += loss.item()
                    count += 1
            
                avg_valid_loss = valid_loss/count

                print("\tTrain Loss = {},\tValidation Loss = {}".format(avg_train_loss, avg_valid_loss))

                self.valid_loss_v_epoch[0][epoch] = epoch
                self.valid_loss_v_epoch[1][epoch] = avg_valid_loss
            else:
                print("\tTrain Loss = {}".format(avg_train_loss))

            self.train_loss_v_epoch[0][epoch] = epoch
            self.train_loss_v_epoch[1][epoch] = avg_train_loss

    def plot_loss(self):
        plt.subplot(121)
        plt.title("Epoch vs train loss")
        plt.plot(self.train_loss_v_epoch[0], self.train_loss_v_epoch[1])

        plt.subplot(122)
        plt.tick_params("Epoch vs validation loss")
        plt.plot(self.train_loss_v_epoch[0], self.train_loss_v_epoch[1])
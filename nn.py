import torch
from torch import nn

import torch.optim.lr_scheduler as lr_scheduler

class Classifier (nn.Module):
    
    def __init__ ( self, p=0.2, layers_nodes=[256,128] ):
        
        super(Classifier, self).__init__()
        
        if p:
            self.input_layer = nn.Sequential(
                nn.Linear(784,layers_nodes[0]),
                nn.Dropout(p=p),
            )
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(784,layers_nodes[0]))
            
        self.hidden_layers = nn.ModuleList()      
        
        for ll in range(1,len(layers_nodes)):
            self.hidden_layers.append(nn.Linear(layers_nodes[ll-1],layers_nodes[ll]))
            if p:
                self.hidden_layers.append(nn.Dropout(p=p))

        self.output_layer =  nn.Linear(layers_nodes[-1],10)
        
        self.activation = nn.SiLU()
        
    def forward (self, x):

        latent_x = []
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.input_layer(x))  
        latent_x.append(x)
        for e, ll in enumerate(self.hidden_layers):
            x = self.activation(ll(x))
            if e < len(self.hidden_layers)-1:
                latent_x.append(x)
        y = torch.softmax(self.activation(self.output_layer(x)),dim=-1)

        return y.reshape(-1,10), latent_x

    


def train_classifier (device, network, trainset, batch_size=100, epochs=50, accuracy_target=0.9, step_scheduler=3, verbose=True):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True )
    
    opt = torch.optim.Adam(network.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_scheduler, gamma=0.5)


    for epoch in range (epochs):
        for x,y in trainloader:
            x=x.to(device)
            y=y.to(device)
            opt.zero_grad()
            y_pred,_= network(x)
            loss = nn.CrossEntropyLoss(reduction='mean')(y_pred,y)
            loss.backward()
            opt.step()

        with torch.no_grad():


            loader = torch.utils.data.DataLoader (trainset, batch_size=len(trainset))

            for x,y in loader:                
                x=x.to(device)
                y=y.to(device)
                y_pred, _= network(x)
                loss_train = nn.CrossEntropyLoss(reduction='mean')(y_pred, y)
                accuracy_train = (torch.argmax(y_pred, dim=1)==y).float().mean()

            if (accuracy_train>0.92 and opt.param_groups[0]['lr']>0.000001):
                scheduler.step()
            if (accuracy_train > 0.92 and trainloader.batch_size<32):   
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True )
            if (accuracy_train > 0.93 and trainloader.batch_size<64):   
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True )
            if (accuracy_train > 0.95 and trainloader.batch_size<256):   
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True )
            if (accuracy_train > 0.96 and trainloader.batch_size<500):   
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True )

            if (accuracy_train>accuracy_target):

                print("End of training at epoch: ", epoch)
                print(f"\t loss: {loss_train:>4f}")
                print(f"\t accuracy: {accuracy_train:>4f}")
                return True
            if (accuracy_train<0.85):
                break
            if (epoch%5==0 and verbose):
                    print("Epoch: ", epoch)
                    print(f"\t loss: {loss_train:>4f}")
                    print(f"\t accuracy: {accuracy_train:>4f}")
    
    print('Training not converged.')
    return False
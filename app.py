#!/usr/bin/env python3

from os import system
import torch
import matplotlib.pyplot as plt
#DATA
hours_a_day = [ 4,   6,   8,  12,  16, 20, 22 ]      #time elapsed to the projects per day (max 24 ours, min != 0)   x = hours_a_day = [0;24]
success     = [ 100, 100, 95, 75,  50, 20, 0  ]      # % syccessfully completed projects                             y = success = [0; 100]

#plt.style.use( 'seaborn-whitegrid' )
#plt.plot( hours_a_day, success, color = 'red',  linestyle = 'solid', linewidth = 1, marker = '.' )
#plt.show()

X = torch.tensor( [ 4,   6,   8,  12,  16, 20, 22 ], dtype = torch.float32 )
Y = torch.tensor( [ 100, 100, 95, 75,  50, 20, 0  ], dtype = torch.float32 )

X = X.reshape( -1, 1 )
Y = Y.reshape( -1, 1 )

inputs  = 1
outputs = 1
rate = 0.0005
epoch = 30_000

model     = torch.nn.Sequential(\
    torch.nn.Linear( 1, 9 ),\
    torch.nn.ReLU(),\
    torch.nn.Linear( 9, 9 ),\
    torch.nn.ReLU(),\
    torch.nn.Linear( 9, 1 ),\

)

loss      = torch.nn.MSELoss( reduction = "mean" )
optimizer = torch.optim.Adam( model.parameters(), lr = rate )
print( model )


for i in range( epoch ):
    for param in model.parameters():
        param.grad = None
    Y_pred = model( X )
    step_loss = loss( input = Y_pred, target = Y )
    step_loss.backward()
    optimizer.step()
    if i % 5_000 == 0: 
        print( f"epoch [{i}], Loss: {step_loss.item():.2f} " )

X1 = torch.tensor( [8], dtype = torch.float32 ).reshape( -1, 1 )
Y1 = torch.tensor( [95], dtype = torch.float32 ).reshape( -1, 1 )
Y_pred1 = model(X1).detach()

X1 = torch.tensor( [7], dtype = torch.float32 ).reshape( -1, 1 )
#Y1 = torch.tensor( [95], dtype = torch.float32 ).reshape( -1, 1 )
Y_pred1 = model(X1).detach()
print( X1, Y_pred1 )

#plt.plot( X, Y_pred.detach().numpy(), color = 'blue', linestyle = 'solid', linewidth = 1, marker = '*' )
#plt.plot( X, Y.detach().numpy(),      color = 'red',  linestyle = 'solid', linewidth = 1, marker = '*' )
#plt.show()


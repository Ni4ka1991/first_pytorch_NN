#!/usr/bin/env python3


import torch

#DATA
#hours_a_day = [ 4,   6,   8,  12,  16, 20, 22 ]      #time elapsed to the projects per day (max 24 ours, min != 0)   x = hours_a_day = [0;24]
#success     = [ 100, 100, 95, 75,  50, 20, 0  ]      # % syccessfully completed projects                             y = success = [0; 100]


X = torch.tensor( [[ 4,   6,   8,  12,  16, 20, 22 ], [ 5, 4, 6, 7, 9, 33, 76 ]], dtype = torch.float32 )
Y = torch.tensor( [ 100, 100, 95, 75,  50, 20, 0  ], dtype = torch.float32 )

print( X )
print( X.shape )
print( Y )
print( Y.shape )





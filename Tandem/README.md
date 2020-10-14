# Tandem model
## This is the Tandem model implemented by Ben in Pytorch, transfering from previous tf version 
# Developer Log:

### RoadMap for this work:
1. Identify the difference between the tandem model with the forward model in implementation in Pytorch
2. Implement them

## 2019.12.05
Function completed:
1. parameter.py and flag_reader commented

## 2019.12.14
Function completed:
1. Forward and backward model separation for Pytorch
2. Saving module modification to cater the separated saving
3. The training on CPU tested with boundary loss implemented and documented on tensorboard
4. Bug fixed for increasing backward training loss
 

# To-do list
1. Evaluation module

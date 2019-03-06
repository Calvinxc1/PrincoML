# tf_modular_nn
My personal modular Neural Net code

## Revision History

#### 2019/03/06
* Controller: Fixed a bug created in the resequenced learning routine, which broke best coef saving, still doesn't save -inf coefs though

#### 2019/03/05
* Learning Rate is now a class within the Dense Learn Module, allowing for dynamic learning rates
* Created the following Learn Rate classes: Flat, Time Decay, Exponential Decay, Step Decay
* Controller: Added learn rate record, and visualizer method for that record
* Smooth Learner: Modified gradient and grad_sq internals to not start at 0, but with the first gradient and grad_sq values
* Controller: Resequenced learning routine so that -inf losses don't override best coefs
* Created noisey gradient learner classes, attached to the Dense Learn Module, currently just None (Root) and Gaussian noise

#### 2019/03/04
* Added reshape to all cluster output
* Altered merge cluster to accept 3D tensors
* Survival Merger: pulling from correct input data; correctly allowing multiple space outputs
* Added Orthogonal initializer, w/ Xavier option
* Norm Regularizer: Added sqrt for L2 regularization, now balances correctly with L1

#### 2019/03/02
* Refactored Survival Merger to include 'prob' and 'sig' space outputs

#### 2019/03/01
* 'best loss' now properly saves coefs at best loss iteration
* Allowed null loss terminate to also terminate on inf loss
* Added sigmoid/probability space output to survival merger

#### 2019/02/28
* SoftRelu Activator: Fixed input_tensor copy operation; Added 'leak'
* Survival Merger: Fixed input_tensor parser operation
* Controller: Added a null loss terminate to learning; Added a loss smoother; Added indexing to plot_losses method; Refactored coefs method; Added a manual data prediction method
* Implemented 'best loss' coef save & recall mechanism


## Design Backlog
* Build a feature importace shuffler (shuffle columns and test error)
* Build out Convolutional learner
* Build out Recurrent/State wrapper for clusters
* Build out a repeating sequence wrapper for clusters
* Add Xavier option to Normal initializer
* Dynamic L1/2 regularization


## Known Bugs
* Some instances of 'best loss' do not visualize the correct iteration of 'best loss' using the plot_losses method in the controller, may also impact which coefs are saved
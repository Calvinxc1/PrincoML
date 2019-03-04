# tf_modular_nn
My personal modular Neural Net code

## Revision History

#### 2019/03/04
* Added reshape to cluster output
* Altered merge cluster to accept 3D tensors
* Bugfix survival merger: pulling from correct input data; correctly allowing multiple space outputs
* Added Orthogonal initializer, w/ Xavier option
* Added sqrt for L2 regularization, now balances correctly with L1

#### 2019/03/02
* Refactored Survival Merger to include multiple space outputs

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
* Build out Recurrent/State wrapper
* Add Xavier option to Normal initializer
* Build a Stochastic Learner module addon
* Develop a diminishing learing rate algorithm
* Dynamic L1/2 regularization


## Known Bugs
* Some instances of 'best loss' do not visualize the correct iteration of 'best loss' using the plot_losses method in the controller, may also impact which coefs are saved
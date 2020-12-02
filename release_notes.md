## Revision History

#### 2020/12/01
* Set up as a 'proper' PyPi package

#### 2019/04/15
* Created a missing data handler class for the data cluster

#### 2019/04/12
* Refactored directory name to match common Python naming conventions

#### 2019/04/11
* Renamed the library to PrincoML
* Added an actual description to the readme

#### 2019/04/08
* Fixed visualization of optimal iteration coefficents on multiple-trained controllers
* Fixed regularization function after coefficent refactor
* Fixed Coefficent locking after coefficent refactor
* Changed Controller Learn lock_coefs kwarg default to True

#### 2019/04/05
* Fixed testbeds, learning rate inputs correctly now
* Refactored learning modules to store coefficents as values in a dict
* Created hinge regression learning module
* Fixed the 'best loss' marker not showing correctly in plot_losses (was a display issue)

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
* Build a feature importace shuffler, based on shuffling columns and examining change in loss
* Build out Convolutional learner
* Build out Recurrent/State wrapper for clusters
* Build out a repeating sequence wrapper for clusters
* Add Xavier option to Normal initializer
* Dynamic L1/2 regularization
* Build out a coefficent variance estimator - added 2019-03-19
* Build out a constructor module to support easy feed-forward NN creation - added 2019-03-19
* Build out a model state snapshoter - Added 2019-04-08
* Add a geometric mean loss combiner (using log adding) - Added 2019-04-12


## Known Bugs
* Still getting some plot_losses render bugs

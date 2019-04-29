# PrincoML
A few years back I got frustrated with the current derth of Machine Learning libraries out there, namely TensorFlow, PyTorch and Keras. They work great using the standard methods, but I wanted to make something more configurable, where I could customize the architecture, equations, and whatnot, in a much more straightforward method. As such I made this library, PrincoML (named after a good friend of mine).

This is a fantastic library if you're wanting to experiment with novel gradient-based approaches to supervised learning. This is a horrible library to use if you want an easy-to-use out-of-the-box approach to machie learning.

PrincoML allows for a level of flexability beyond what existing libraries provide in how a gradient-descent based machine learning solution can be architectured. It supports simple linear/logistic/GLM style regression, as well as fantastically complex neural networks, and all sorts of solutions inbetween. I've used it both on personal projects and professionally, and it's worked wonders for me. PrincoML is based on PyTorch as the backend, which is a personal preference, as I've tried both Numpy and TensorFlow impementations, and PyTorch just offers a smoother integration for the needs of the library.

Development is largely based around whatever feature I need or feel like implementing. This likely means that CNN/Image Recognition stuff will take a back seat as I'm personally not fond of those approaches, and RNN/Time Series/NLP stuff will take forefront, as that's my speciality (-:

I'm always welcome to accept ideas/code/etc. from other contributors, but the primary purpose of this library is to serve as my own personal machine learning code, that I just happen to share with everyone else (-: As such, enjoy, use at your leisure, and have fun with it!

- Jason Cherry

<JCherry@gmail.com>


## Revision History

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

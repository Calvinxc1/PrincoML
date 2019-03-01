# tf_modular_nn
My personal modular Neural Net code

## Revision History

#### 2019/02/28
* SoftRelu Activator: Fixed input_tensor copy operation; Added 'leak'
* Survival Merger: Fixed input_tensor parser operation
* Controller: Added a null loss terminate to learning; Added a loss smoother; Added indexing to plot_losses method; Refactored coefs method; Added a manual data prediction method
* Multiple: Implemented 'best loss' coef save & recall mechanism

## Design Backlog

* Build a feature importace shuffler (shuffle columns and test error)
# hypernetRegularisation

There are two parts to this repo: 
1. The orig_* part : it's part for the original experiment with no possibility to smooth the regularization. Uploaded this part because I had a problem with the other part (the one where smoothing is possible) and I had no idea if I would be able to fix or not. In the models folder, there are the scripts for the three parts of these experiments (they are supposed to work with CIFAR10 or CIFAR100).
The orig_train.py part is the script from which you can launch the experiments using python orig_train.py. You can control the setup or the experiments using the command line arguments (whether to do dropout or not, which layer to regularise in the ResNet, whether you want to compare two models ...). Keeps in the end 5 different plots: one for the accuracy on the train, one for accuracy on the test, loss every 50 batches and the difference betwenn the train and test accuracy, one for the number of non zero parameters passed to the primary and a json of the results.
2. The upgraded part: you can begin to do some basic smoothing to the regularization (namely choose from which epoch to regularize). 
You can launch experiments with this part using python main.py, and again you can control the setup of the experiment using command line arguments. Keeps in the end 4 different plots: one for the accuracy on the train, one for accuracy on the test, loss every 50 batches and the difference betwenn the train and test accuracy.

!!!! This repo is still young, and under work. Many upgrades will be pushed in the weeks to come !!!!

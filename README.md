# Graduate Project

## 1. Preprocess Data

We strongly recommend you use Google Colab to complete this part preparation. All codes are in dirflower/src/py/data_preprocess

For Cora, you just need to prepare data and use code in pyMetis_cora.

For PubMed, you have to use pubmed_preprocess first. Then you can use pymetis_med to partition the data.

## 2. Experiment

To activate the experiment, after deployment, we need to get IP list of all pods. Change the IP list in the script and then copy_key.sh to copy the ssh key to the pods. 

Then all the preparation jobs have finished, and we shall begin our experiment.

The parameters of those scripts are described at the beginning of those scripts. Here are some instructions for parameters;

fraction_fit: static 0.1

fraction_eval: static 0.1

min_fit_clients:  In synchronous strategy, this means the minimum training client number in each round. 

min_eval_clients: static 12

Rounds: the number of rounds of FL

min_available_clients: static 12

Alpha: in asynchronous strategy, means the static alpha value. In adaptive alpha, means the basic alpha value.

Staleness: in adaptive alpha, means the threshold of staleness.

Strategy: 0 asynchronous fl; 2 adaptive alpha

Log_n: the log file name of this experiment.

e: how many epochs should the client train in each round

Lr: the learning rate of the client during the training process

After activating the experiment, the accuracy of each round will be recorded in the result file. Then we can use the visualization script to visualize the result.

If you want to run synchronous experiments, use sync.sh and choose suitable parameters.

If you want to run asynchronous experiments, use async.sh and choose suitable parameters.

# FreeLunchSemiParametrics


A testbed for comparing models with and without influence function update steps, accompanying paper: https://openreview.net/forum?id=dQxBRqCjLr


Q (outcome) methods:
- Linear/Logistic Regression
- SuperLearner (van der Laan, Polley & Hubbard, 2007)
- CFR (Shalit, Johansson & Sontag, 2017)
- MultiNet (Vowels, Akbari, Camgoz & Bowden, 2023)
- TVAE (Vowels, Camgoz & Bowden, 2021)
- DragonNet (Shi, Blei & Veitch, 2019)
- S-Learner (Kunzel, Sekhon, Bickel & Yu, 2019)
- T-Learner (Kunzel, Sekhon, Bickel & Yu, 2019)

G (propensity) methods:
- Linear/Logistic Regression
- SuperLearner (van der Laan, Polley & Hubbard, 2007)
- CFR (Shalit, Johansson & Sontag, 2017)
- P-Learner (Zou & Hastie, 2005)
- DragonNet (Shi, Blei & Veitch, 2019)

U (update) methods:
- OneStep / von Mises expansion method
- Submodel / targeted update (van der Laan & Rose, 2011)
- MultiStep + variations (Vowels, Akbari, Camgoz & Bowden, 2023)
- Targeted Regularization (Shi, Blei & Veitch, 2019; Vowels, Camgoz & Bowden, 2021)

Other:
- Double Machine Learning (Chernozhukov et al., 2018)



We also include code for automatically deriving an expression for the influence function in 
```auto_IF.py``` with an example in ```autio_IF.ipynb``` [Special thanks to Sina Akbari for this work]


## Installation

The installation process is slightly involved owing to the use of implementations from multiple libraries...

### First install the CausalML package from https://github.com/uber/causalml

These instructions are modified from those on the original method's github (the ones on github did not work directly)
```
git clone https://github.com/uber/causalml.git  
cd causalml/envs/
conda create -n causal-ml python=3.7
conda activate causal-ml
cd ..
pip install -r requirements-tf.txt
pip install numpy==1.20.1
python setup.py build_ext --inplace
python setup.py install
```

N.B. The tested folder structure clones causalml to the current working directory (FreeLunchSemiParametrics).


Install the doubleml (double machine learning) package if you wish to evaluate their method https://docs.doubleml.org/stable/index.html it will get called in Q_main.py
which can be run using the causal_ml environment.


Then modify ```utils.py``` from causalml.inference.tf.utils (probably located @ ```...envs/causal-ml/lib/python3.7/site-packages/causalml-0.11.1-py3.7-linux-x86_64.egg/causalml/inference/tf/utils.py``` ) to include two new functions:
```python
def dragonnet_loss_binarycross_cat(concat_true, concat_pred):

    return binary_outcome_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)
```
and

```python
def binary_outcome_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pred = t_true * y1_pred + (1. - t_true) * y0_pred
    y_pred = tf.keras.activations.sigmoid(y_pred)
    y_pred = (y_pred + 0.001) / 1.002
    lossy = tf.reduce_sum(K.binary_crossentropy(y_true, y_pred))

    return lossy
```

These functions deal with problems when running DragonNet with categorical outcomes.

Also, the code here: https://github.com/uber/causalml/blob/master/causalml/inference/tf/dragonnet.py (lines 220-228),and  here: https://github.com/claudiashi57/dragonnet/blob/master/src/experiment/ihdp_main.py (lines 88-102)
involves a redundant optimization step (first with Adam, then using SGD). Checking in the original paper here: https://arxiv.org/abs/1906.02120 the authors say (page 8) that SGD works best, which corresponds with the second optimizer. ##

However, whenever we used the SGD optimizer we ran into problems with NaN weights when using the targeted regularization. We therefore removed the SGD
optimizer and use only the Adam optimizer.


If you wish to do the same thing: modify ```dragonnet.py``` from causalml.inference.tf (probably located @ ```...envs/causal-ml/lib/python3.7/site-packages/causalml-0.11.1-py3.7-linux-x86_64.egg/causalml/inference/tf/dragonnet.py``` )


### Then install the my_gpu_torch environment (you can use the requirements.yml for this if you want)

```bash
conda create -n my_gpu_torch python=3.7 
pip install causaleffect=0.0.2
conda install matplotlib=3.2.2 pandas=1.2.5 scikit-learn=0.24.2 seaborn=0.11.0 statsmodels=0.12.2 tqdm=4.62.3 scipy=1.6.2 notebook=6.4.6 numpy=1.20.3 pillow=8.4.0 pytorch=1.10.0
conda install -c pytorch torchaudio=0.10.0 torchvision=0.11.1
conda install pyro-ppl=1.7.0
```


### Modify paths
In ```main.sh``` there are some paths for anaconda envs etc. which you will need to change to suit your system.



## Experiments

Give ```main.sh``` file permissions if necessary using e.g. ```chmod +x main.sh``` and run ```main.sh```.

You can specify which dataset, and which Q and G models you wish to run in ```main.sh```.



## Evaluating Results

An example jupyternotebook for accessing results is given ```in results_exploration-DML.ipynb```. This script
is designed to access the results for the aeATE comparison, because the DML approach does not provide individual
per-participant predictions, but just an overall estimate of the ATE.



## Trouble Shooting

1. If you get the error ```ModuleNotFoundError: No module named 'typing_extensions'``` the solution which has worked (for me) is 
to <activate the relevant conda environment>, and then run:

```pip install typing_extensions ```


2. If you get the error: Test_NNLS.test_nnls and TestNNLS.test_maxiter fails with missing n
try solution from here:

https://github.com/scipy/scipy/issues/12273 

The ugly solution is to find the ```__init__.py``` for ```scipy.optimize``` in the ```site-packages``` directory for the environment, and change the import: 
```from _nnls import nnls``` to  ```from .nnls import nnls```


3. If you get the error saying that the classifier has no method predict_proba, this is because sklearn's stackingclassifier
only uses the predict_proba on the output learner, and this hassattr() check does not identify it. The (again ugly) solution
is to delete lines 948-954 of double_ml.py which check for this method (and fail to fined it). 

## Auto-IF
The code for identification and derivation of the influence function is demonstrated in ```auto_IF.ipynb```.




### Teething Issues / TODOs:
Sometimes the experiments fail if, during one of the cross-validation loops, there exist only participants in one but not the other 
treatment groups. This tends to occur in datasets with close-to-positivity violations. For now, one has to restart the script on
a different simulation.


## References

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/debiased machine learning for treatment and structural parameters. Econometrics Journal, 21:C1–
C68, 2018.

S. R. Kunzel, J.S. Sekhon, P.J. Bickel, and B. Yu. Meta-learners for estimating heterogeneous treatment
effects using machine learning. arXiv preprint, arXiv:1706.03461v6, 2019.

U. Shalit, F. D. Johansson, and D. Sontag. Estimating individual treatment effect: generalization bounds
and algorithms. ICML, 2017.

C. Shi, D. M. Blei, and V. Veitch. Adapting neural networks for the estimation of treatment effects. 33rd
Conference on Neural Information Processing Systems, 2019.

M.J. van der Laan, E.C. Polley, and A.E. Hubbard. Super Learner. Statistical Applications of Genetics and
Molecular Biology, 6(25), 2007. doi: 10.2202/1544-6115.1309.

M. J. van der Laan and S. Rose. Targeted Learning - Causal Inference for Observational and Experimental
Data. Springer International, New York, 2011.

M. J. Vowels, N.C. Camgoz, and R. Bowden. Targeted VAE: Structured inference and targeted learning for
causal parameter estimation. IEEE SMDS, 2021.

M.J. Vowels, S. Akbari, C. Camgoz, and R. Bowden. A Free Lunch with Influence Functions? An Empirical Evaluation of Influence Functions for Average Treatment Effect
Estimation. TMLR, 2023. https://openreview.net/pdf?id=dQxBRqCjLr

H. Zou and T. Hastie. Regularization and variable selection via the elastic net. J. R. Statist. Soc., 67(2):
301–320, 2005.


### Key Libraries We Use / Reference:

https://github.com/uber/causalml

https://docs.doubleml.org

https://github.com/pedemonte96/causaleffect

https://pyro.ai/

https://github.com/matthewvowels1/TVAE_release

https://github.com/claudiashi57/dragonnet
set -e

for i in {1..100}  # number of simulations
do
# choose DATASET to use (options include synth1, synth2, general and IHDP). The synth datasets are variants of the Luque-Fernandez et al. 2018 dataset.
# The options in the paper are synth1 = LF (V1), synth2 = LF (V2), IHDP, and general ('Gen' in the article).
DATASET="general"
RUN="RUN_all_general_TEST"  # name of run
N=5000  # sample size (only affects synth datasets, IHDP is fixed at N=747)
ITER=$i

echo "SIM $i DATASET $DATASET N $N"
# Choose which Q models to evaluate:
TRAINQLR=1  # Logistic Regression Q
TRAINQS=1  # S Learner Q
TRAINQT=1  # T Learner Q
TRAINQD=1  # DragonNet learner Q
TRAINQSL=1  # SL learner Q
TRAINQCFR=1  # CFR learner Q
TRAINQMN=1   #  MultiNet learner Q
TRAINQTVAE=1   # TVAE
TRAINQDML=1   # Double ML


# Choose which G models to evaluate:
TRAINGCFR=1   # CFR learner G
TRAINGMN=1   # MultiNet Learner G
TRAINGSL=1   # SL learner G
TRAINGLR=1   # LR learner  G
USEP=1    # P learner G

# EVAL FILENAMES
FN_="data/"
FN_MO="model_output/"

# Debugging:
RUNA=1
RUNB=1
RUNC=1
RUND=1

WDIR="$(dirname "$0")"

if [ $RUNA -eq 1 ]
then
echo "============= A: Generating Data... ============="
source /opt/conda/bin/activate my_gpu_torch
/opt/conda/envs/my_gpu_torch/bin/python "$WDIR/helpers/data_gen_main.py" --dataset $DATASET --run $RUN --N $N --iteration $ITER
conda deactivate
fi


if [ $RUNB -eq 1 ]
then
if [ $TRAINQCFR -eq 1 ] || [ $TRAINQMN -eq 1 ] || [ $TRAINQTVAE -eq 1 ]
then
# B.1 Train Q CFR and MultiNet models and save model outcome predictions
echo "============= B: Training models (such as) CFR and MN... ============="
source /opt/conda/bin/activate my_gpu_torch
/opt/conda/envs/my_gpu_torch/bin/python "$WDIR/helpers/Q_main_mncfr.py" --dataset $DATASET --run $RUN --train_mnlearner $TRAINQMN --train_cfrlearner $TRAINQCFR --iteration $ITER --N $N --train_tvae $TRAINQTVAE
conda deactivate
fi

# B.2 Train Q models using causalml package and save model outcome predictions (some models also output propensity scores)
source /opt/conda/bin/activate causal-ml

# Get the current TensorFlow version
current_version=$(python -c "import tensorflow as tf; print(tf.__version__)")

# Define the minimum required version of TensorFlow
minimum_version=2.4

# Compare the versions using the python package "packaging"
if python -c "import packaging.version; exit(0) if packaging.version.parse('$current_version') >= packaging.version.parse('$minimum_version') else exit(1)"; then
    echo "TensorFlow is already up-to-date. (Current version: $current_version)"
else
    # Upgrade TensorFlow
    conda install tensorflow-gpu
    echo "TensorFlow upgraded to the latest version! (Previous version: $current_version, New version: $(python -c "import tensorflow as tf; print(tf.__version__)"))"
fi

echo "============= B: Training alternative models (such as) DragonNet, SL etc... ============="
/opt/conda/envs/causal-ml/bin/python "$WDIR/helpers/Q_main.py" --dataset $DATASET  --N $N  --run $RUN --train_dmllearner $TRAINQDML --train_tlearner $TRAINQT --train_slearner $TRAINQS --train_dragon $TRAINQD --train_sllearner $TRAINQSL --train_lrlearner $TRAINQLR  --iteration $ITER
conda deactivate
fi



if [ $RUNC -eq 1 ]
then
echo "============= C: Training Propensity Score Models... ============="
source /opt/conda/bin/activate causal-ml
# C. Train G and save model propensity score predictions
/opt/conda/envs/causal-ml/bin/python "$WDIR/helpers/G_main.py" --dataset $DATASET --run $RUN --N $N  --train_lr $TRAINGLR --train_sl $TRAINGSL --train_mn $TRAINGMN --train_cfr $TRAINGCFR --iteration $ITER
conda deactivate
fi


if [ $RUND -eq 1 ]
then
# D. Evaluate models with and without targeted/IF update(s)
echo "============= D: Evaluating Models... ============="
source /opt/conda/bin/activate my_gpu_torch
/opt/conda/envs/my_gpu_torch/bin/python "$WDIR/helpers/eval_main.py" --dataset $DATASET --QDML $TRAINQDML --QTVAE $TRAINQTVAE --fn $FN_ --fn_mo $FN_MO --run $RUN --QS $TRAINQS  --QT $TRAINQT --QMN $TRAINQMN --QCFR $TRAINQCFR --N $N  --QLR $TRAINQLR --QSL $TRAINQSL --QD $TRAINQD --GLR $TRAINGLR --GSL $TRAINGSL --GMN $TRAINGMN --GCFR $TRAINGCFR --GP $USEP --GDP $TRAINQD --iteration $ITER
conda deactivate
fi
done
echo "============= All Done! :] ============="
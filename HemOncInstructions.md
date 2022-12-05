# Evaluating The System on the HemOnc Data

## Disclaimer

The annotated HemOnc data is a small dataset, and is not a representative sample of all of our whole training data.
This dataset is provided for demonstration purposes only,
and is not meant to reproduce the results in the paper. 

If you use this dataset to train on train, optimize on dev, and evaluate on the test split, the results you get will be low (see the tables below). If you get these (approximate) results, it means you have used the code correctly.

## Model Training

You can run the training scripts from `Training_Scripts` at the top level of the repository.
After activating your relevant conda environment,
run the scripts from the top level of the repository directly, e.g.:
```
sh Training_Scripts/NER/boost.sh
```
etc. for the NER models and:
```
sh Training_Scripts/RE/inst_rel.sh
```
for relations.  The `boost.sh` script consists of:
```
task=rt_boost
dir="$(pwd)/HemOnc_RT_Data/NER/Boost"
epochs=10
lr=2e-5
ev_per_ep=2
stl=0
seed=42
gas=4
lr_type=constant
encoder="emilyalsentzer/Bio_ClinicalBERT"
dmy=$(date +'%m_%d_%Y')
#logging_dir="/home/$USER/step_recovery/logs/$task/$encoder/ep=$epochs-lr=$lr-stl=$stl-seed=$seed-gas=$gas/"
#mkdir -p $logging_dir
#logging_file="$logging_dir/$dmy"
#touch $logging_file
temp="/home/$USER/step_recovery/$task/$encoder/ep=$epochs-lr=$lr-stl=$stl-seed=$seed-gas=$gas/$dmy/"
cache="/home/$USER/step_recovery/caches/$task/$encoder/ep=$epochs-lr=$lr-stl=$stl-seed=$seed-gas=$gas/$dmy"
mkdir -p $cache
mkdir -p $temp
python -m cnlpt.train_system \
--task_name $task \
--data_dir $dir \
--encoder_name $encoder \
--do_train \
--cache $cache \
--output_dir $temp \
--overwrite_output_dir \
--evals_per_epoch $ev_per_ep \
--do_eval \
--num_train_epochs $epochs \
--learning_rate $lr \
--lr_scheduler_type $lr_type \
--seed $seed \
--gradient_accumulation_steps $gas #>>$logging_file 2>&1
```

We use the same hyperparameters as in the paper results in the scripts by default.


By default the training experiments are organized
in `~/step_recovery/<task name>/<encoder name>/<hyperparameters>/<experiment date>`.
You can change the storage (and any logging directories) via the script.
The `temp` directory is where the model checkpoints and results are saved,
`cache` is for the data converted for model input.  

The best model with its needed information for running in pipeline mode
will be the `pytorch_model.bin` file etc. at the top level of that directory.  

## Finding Scores and Best Epoch Fragments

At the top level of the aforementioned directory there will be a file called `eval_results.txt`.
This will contain a trace of the model's evaluations on the dev set,
and at the end, a summary of the evaluation checkpoint for the best model,
i.e. best score, best step, best epoch fragment.

Our results for the individual models were obtained on
the same `conda` environment as you can set up in the main README,
on a workstation with a GeForce RTX 3080 Ti, running Ubuntu 22.04 LTS.
When running this code your results should be similar.

*NB:* We do **not** train a model for NER date detection (`rt_date`),
since there are no instances in HemOnc.

### NER Results

| Task Name | F1   | Best Epoch  |
|-----------|------|---------------------|
| Boost     | 0.95 | 2                   |
| Dose      | 0.99 | 7.2                 |
| FxFreq    | 0.89 | 18                  |
| FxNo      | 0.96 | 13                  |
| Site      | 0.65 | 2.5                 |

### RE Results

Best epoch: 18

|    | None | Dose-Boost | Dose-Dose | Dose-FxFreq | Dose-FxNo | Dose-Site | All Categories |
|----|------|------------|-----------|-------------|-----------|-----------|----------------|
| F1 | 0.98 | 0.86       | 0.83      | 0.88        | 0.9       | 0.7       | 0.83           |

## Running The Pipeline

Create a folder (I used `~/Pipeline_HemOnc_Dev`) and make a subfolder for each task name,
and move the contents from the top level of each training folder: 
```
~/step_recovery/<task name>/<encoder name>/<hyperparameters>/<experiment date>
```
to the corresponding task subfolder of `~/HDD/Dev_Hemonc_Pipeline`. 

Once you have the models in their appropriate locations,
you can run the pipeline via the command (using the prior example direcotries):

```
python -m cnlpt.cnlp_pipeline --models_dir ~/Pipeline_HemOnc_Dev/ --axis_task rt_dose --in_dir <personal_cnlpt location>/HemOnc_RT_Data/E2E/filtered_dev/ --out_dir ~/cnlpt_predictions --mode eval
```
This will print output files with metrics and predictions to the provided `out_dir` argument,
in this case `~/cnlp_predictions`

In our environment we obtain the following scores
```
Score              None    DOSE-BOOST    DOSE-DOSE    DOSE-FXFREQ    DOSE-FXNO    DOSE-SITE
---------  ------------  ------------  -----------  -------------  -----------  -----------
f1             0.997921             0     0.289855              0            0     0.189655
recall         0.997073             0     0.689655              0            0     0.297297
precision      0.998769             0     0.183486              0            0     0.139241
support    27678                    7    29                    22           51    37

acc: 0.9929557216791259
Gold classes missing in predictions (whole class false negative): ['DOSE-FXFREQ', 'DOSE-FXNO']
```

## Error Analysis

In some example output from the predictions file we can get a better idea of where in the pipeline we are having issues:


```
6.

Relation type counts:
( DOSE-DOSE , 3 )

Predicted positive labels:

Model Predicted Labels:

( 8 , 16 , DOSE-DOSE) -> ( 25 , 50 Gy )

Labels Inferred From Predictions:

( 4 , 8 , DOSE-DOSE) -> ( 2 Gy , 25 )
( 4 , 16 , DOSE-DOSE) -> ( 2 Gy , 50 Gy )



Ground truth paragraph:

<cr> <cr> <cr> <cr> 2 Gy fractions x 25 fractions , for a total dose of 50 Gy , to start within 4 hours after the first dose of chemotherapy .

Discovered Entities

Anchor mentions:	Signature mentions:

( 4 , 5 , 2 Gy, source : rt_dose )	( 8 , 8 , 25, source : rt_fxno )
( 16 , 17 , 50 Gy, source : rt_dose )			

```

Looking at the corresponding ground truth labels in the filtered HemOnc dev split
```
(4,8,DOSE-FXNO) , (4,16,DOSE-DOSE) , (8,16,DOSE-FXNO)
```

We can see that that the pipeline has generated relations at the correct indices,
but with incorrect labels.

We can see from the `Discovered Entities` column that while each of the entity spans
was appropriately classified,
i.e. the relation classifier recieved the equivalent of gold data.
However as we can see from the output,
(4, 8) and (8, 16) were misclassified as DOSE-DOSE instead of DOSE-FXNO.

We can infer from the 'predicted' vs. 'inferred' labels,
that there were two predicted DOSE-DOSE labels (without loss of generality)
((4, 8) and (8, 16)),
and that this pair generated the (4, 16)
DOSE-DOSE during the transitive closure generation stage.
However during the next linking stage the newly discovered
(4, 16) link as well as the (4, 8) generated a new (8, 16)
with a stronger score than the original (since it inherits the strongest score from its generating pair),
thus the appearence of two 'inferred' DOSE-DOSE's from one 'predicted'.

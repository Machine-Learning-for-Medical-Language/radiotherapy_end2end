# An end-to-end natural language processing system for automatically extracting radiotherapy events from clinical texts

## License

This code is released under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Paper

The paper for which this code was written is [here]( https://doi.org/10.1016/j.ijrobp.2023.03.055 ).
This README does *not* include information on the schema or guidelines used in annotating the data,
or more than basic information about the source of the data.
For technical questions not pertaining to this code, please see the paper.

## Description

The provided code and data comprises a system for extracting radiotherapy treatment events from clinical text,
using paragraph parsing from [cTAKES](https://ctakes.apache.org),
followed by event extraction over the paragraphs using
[transformer](https://huggingface.co/docs/transformers/main/en/index)-based
named entity recognition (NER) and relation extraction (RE) models, and rule-based post-processing.

Please note that the system uses individual NER models for each entity, and a single multi-class RE model for all relations.

## User Community

Physicians, cancer registries, or individual developers,
interested in extracting radiotherapy treatment events from free form clinical text.

## Usability

The provided code can be used to train a model and classify it on the provided data in the `HemOnc_RT_Data` folder
(or other data the developer has access to).
The provided code uses data from [HemOnc.org](HemOnc.org), the Kentucky Cancer Registry, and the Mayo Clinic,
that has been downloaded, annotated, paragraph parsed, and converted to `tsv`.

There are three types of `tsv` data formatting used in this code, one for training named entity recognition NERmodels,
one for training the RE model,
and one for evaluating the full end to end pipeline system.
For further explanation see the [data format section](#input-format).   

We include the formatted training and evaluation data from HemOnc.org,
and not from the other two sources, due to data use agreements and to protect patient privacy.
For this reason we do not include our models used in the paper,
however we include results from models trained on the included data for reproducibility.  

## Uniqueness

Radiotherapy treatment is often only described in clinical documents, but end-to-end extraction of these treatments from text is an as yet
unaddressed problem in clinical natural language processing.
The authors hope this work will contribute to cancer surveillance and outcomes research in general by facilitating real-world evidence generation. Future goals including using these methods to help construct comprehensive cancer treatment timelines.

## Point of Contact

If you have trouble running this software or find any issues, please contact [Eli Goldner](mailto:eli.goldner@childrens.harvard.edu?subject=[GitHub]%20RT%20System%20Code). 

## Process Overview

To reproduce our steps:
- [Install this code](#installation)
- [Train models for each RT task](#training-models)
- [Run the models in pipeline mode on the data](#pipeline-evaluation)


The pipeline evaluation provides both instance averaged and document averaged results.
The models will provide instance averaged results on a validation set
(just rename the test split or any other file to `dev.tsv` to have the model evaluate on that).
If however you have data organized by document and want to obtain document averaged results for an
individual model, see [this section](#individual-model-document-level-evaluation).


## Installation

```
conda create -n cnlp python=3.8
```

Follow the instructions for installing PyTorch for CUDA [here](https://pytorch.org/get-started/locally/).
If you do not have GPU access, this code can *in theory* train models and load them for inference on CPU with significant RAM,
however we do not recommend this nor do we support it at this time.
If this is your use case or you require other optimizations
see the [Huggingface pipeline documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
for possible strategies for inference, for training optimizations see [Huggingface's suggestions](https://huggingface.co/docs/transformers/perf_train_cpu).   


Download the repository via either:
- HTTPS
```
git clone https://github.com/Machine-Learning-for-Medical-Language/radiotherapy_end2end.git 
```
- SSH
```
git clone git@github.com:Machine-Learning-for-Medical-Language/radiotherapy_end2end.git
```
Then  

```
cd radiotherapy_end2end && pip install -e .
```
will install the package along with all the non-PyTorch dependencies.

## Input format

### Named Entity Recognition

(Examples from [HemOnc.org](https://hemonc.org/wiki/Main_Page))

Newline separated tab delimited (`tsv`) files.

Tags are one of `O`, `B-<mention type>`, `I-<mention type>`, respectively, the token is outside of a mention, the token is the beginning of a mention, and the token is inside of a mention.

label:
>`O O O B-Anatomical_site O B-Anatomical_site I-Anatomical_site I-Anatomical_site O B-Anatomical_site I-Anatomical_site I-Anatomical_site I-Anatomical_site I-Anatomical_site O O O O O O B-Anatomical_site I-Anatomical_site O`


tokenized cTAKES paragraph:
>`Then 5.4 Gy boost to secondary target volume of 1 - to 1.5-cm margin on all sides , including proven nodal involvement .`

in the actual files these are separated by a tab, presentation here is for readability.

### Relation Extraction

Tokens which comprise the anchor dose mention are surrounded by the `<RT_DOSE-START>` and `<RT_DOSE-END>` tags.  Tokens which comprise the attribute mention (one of boost, date, second dose, fraction number, fraction frequency, site) are surrounded by their corresponding tags, e.g. `<BOOST-START>, <BOOST-END>`, `<DATE-START>, <DATE-END>`, `<DOSE-START>, <DOSE-END>`, etc.   

label:
>`DOSE-DOSE`


cTAKES paragraph:
>`of 30 Gy to point A given in 5 fractions , starting week 4 of XRT <RT_DOSE-START> 1.8 Gy <RT_DOSE-END> fractions x 28 fractions , for a total dose of <DOSE-START> 50.4 Gy <DOSE-END> . The last 5.4 Gy of the 50.4 Gy is limited to the tumor bed . 1.8 Gy fractions x 25 fractions , then a 5.4 Gy final boost , for a total dose of 50.4 Gy , starting within 24 hours of start of chemotherapy 180 cGy x 22 with 3 cm margin to GTV then 180 cGy x 6 with 2 cm margin to GTV , total 50.4 Gy over 6`

## Training Models

Defining tasks is accomplished in `src/cnlpt/cnlp_processors.py`.
NER tasks are cases of `tagging` tasks in the framework,
relation extraction with provided entities are `classification` tasks.

Here is the script used to train the boost NER model

`boost_train.sh` =

```
task=rt_boost
dir="/home/$USER/RT_cnlpt_data/NER/boost/"

epochs=10
#actual_epochs=1.497041420118343
lr=2e-5
ev_per_ep=0
stl=0
seed=42
gas=4
lr_type=constant
encoder="emilyalsentzer/Bio_ClinicalBERT"
dmy=$(date +'%m_%d_%Y')
temp="/home/$USER/RT_pipeline_models/$task/"
cache="/home/$USER/RT_pipeline_models/caches/$task"
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
#--actual_epochs $actual_epochs \
--learning_rate $lr \
--lr_scheduler_type $lr_type \
--seed $seed \
--gradient_accumulation_steps $gas

```

**NB: `actual_epochs` vs `num_train_epochs`**

The need for these distinct hyperparameters arises from the cases of the cosine or linear learning rate schedulers.
The frequency of the cosine function, or the decay rate of the linear function, that generates learning rate values is defined in terms of the maximum number of epochs given to the model.
If you are using some form of early stopping, as we did, then to properly reproduce one's results one needs to stop the model not just at the
epoch fraction at which it stopped but with the same trajectory of the learning rate scheduler throughout.
Thus where the `actual_epochs` argument is specified, the model's learning rate scheduler will generate a function with a frequency
determined by `num_train_epochs` (the projected number of epochs used in training) but the stopping point will be determined by `actual_epochs`.

## Individual Model Document Level Evaluation

To obtain document averaged scores for a model over a director of `tsv` files:

>```python -m cnlpt.rt_model_eval --model_dir test_pipeline_models/rt_site/ --task_name rt_site --in_dir DocNERcnlpt/site/HemOnc/```

## Pipeline Evaluation

>```python -m cnlpt.cnlp_pipeline --models_dir test_pipeline_models/ --axis_task rt_dose --in_dir kcr-naacr/ --mode eval```
Assumes `models_dir` is of the form:
```
/test_pipeline_models/
├── rt_boost
├── rt_date
├── rt_dose
├── rt_fxfreq
├── rt_fxno
├── rt_rel
└── rt_site

```
And that each task-named subdir is of the form:
```
/rt_boost/
├── added_tokens.json
├── config.json
├── eval_predictions.txt
├── eval_results.txt
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── training_args.bin
└── vocab.txt
```
i.e. contains the actual trained model binary (`pytorch_model.bin`) along with the other files related to the model's configuration.

## Reproducing Results

As explained [earlier](#usability), we cannot provide our trained models or full training data.
However we have provided data from [HemOnc.org](HemOnc.org) for training NER and RE models,
instructions for training models and obtaining end-to-end results with trained models,
and own results from this process using the provided HemOnc.org data.

For the instructions and our results obtained from the same process, please see the file `HemOncInstructions.md` at the top level of this directory.

## Acknowledgements

The work was supported by the US National Institutes of Health (grants UH3CA243120, U24CA248010, R01LM010090, R01LM013486, 5R01GM11435).

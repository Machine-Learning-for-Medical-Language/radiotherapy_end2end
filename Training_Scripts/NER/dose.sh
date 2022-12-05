task=rt_dose
dir="$(pwd)/HemOnc_RT_Data/NER/Dose/"
epochs=10
lr=2e-5
ev_per_ep=2
stl=0
seed=42
gas=4
lr_type=constant
encoder="emilyalsentzer/Bio_ClinicalBERT"
dmy=$(date +'%m_%d_%Y')
logging_dir="/home/$USER/step_recovery/logs/$task/$encoder/ep=$epochs-lr=$lr-stl=$stl-seed=$seed-gas=$gas/"
mkdir -p $logging_dir
logging_file="$logging_dir/$dmy"
touch $logging_file
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

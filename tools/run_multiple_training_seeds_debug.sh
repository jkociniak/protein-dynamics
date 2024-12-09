seeds=(69 420 777 1337)

# run training for each hidden dim and seed
for seed in "${seeds[@]}"
do
    echo "Running training for seed $seed"
    python train_toy_mc.py --training_seed $seed --accelerator cpu --max_epochs 300 --log_dir experiments/lightning_logs
done



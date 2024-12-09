seeds=(42 69 420 777 1337)
hidden_dims=(8)

# run training for each hidden dim and seed
for hidden_dim in $hidden_dims
do
    echo "Running training for hidden dim $hidden_dim"
    for seed in $seeds
    do
        echo "Running training for seed $seed"
        python train_toy_mc.py --encoder_hidden_dim $hidden_dim --training_seed $seed --accelerator cpu --max_epochs 100 --log_dir experiments/lightning_logs
    done
done


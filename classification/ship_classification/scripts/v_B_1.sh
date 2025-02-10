#!/bin/bash

BATCH_SIZES=(64 16 24 32 8)
LEARNING_RATES=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3)
SEEDS=(42 2 34)
PRECISIONS=(32)

for BS in "${BATCH_SIZES[@]}"
do
    for LR in "${LEARNING_RATES[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            for PRECISION in "${PRECISIONS[@]}"
            do
                # Format the learning rate to match the file naming convention
                FORMATTED_LR=$(printf "%.4g" $LR)
                
                # Check if the model file already exists
                MODEL_FILE="/Data_large/marine/Dom/Projects/E2E_data_analysis/classification/results_roberto_2cat_80Tr20TestR/venus/1/model/fp32_venus_seed${SEED}_classifier_${BS}_${FORMATTED_LR}_model.pth"
                if [ -f "$MODEL_FILE" ]; then
                    echo "Skipping BS=$BS LR=$LR SEED=$SEED PRECISION=$PRECISION because model file already exists"
                    continue
                fi
                
                # Run the training script if the model file does not exist
                echo "Running training script with BS=$BS, LR=$LR, SEED=$SEED, PRECISION=$PRECISION"
                python /Data_large/marine/Dom/Projects/E2E_data_analysis/classification/runscripts_venus_roberto/v_B_1.py --batch-size $BS --learning-rate $LR --precision $PRECISION --seed $SEED
            done
        done
        # Aggregate results for this combination of batch size and learning rate
        for PRECISION in "${PRECISIONS[@]}"
        do
            echo "Aggregating results for BS=$BS, LR=$LR, PRECISION=$PRECISION with seeds ${SEEDS[@]}"
            python /Data_large/marine/Dom/Projects/E2E_data_analysis/classification/runscripts_venus_roberto/v_B_1_agg.py --batch-size $BS --learning-rate $LR --seeds "${SEEDS[@]}" --precision $PRECISION
        done
    done
done

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

python plant_village_model.py -s models/akidanet_plant_village_untrained.h5

python plant_village_train.py -l models/akidanet_plant_village_untrained.h5 -s models/akidanet_plant_village.h5 -e 10 -lr 1e-3 $DATA_ARG
python plant_village_eval.py -l models/akidanet_plant_village.h5 $DATA_ARG

# 8-bit quantization and tuning
cnn2snn quantize -m models/akidanet_plant_village.h5 -i 8 -w 4 -a 4
python plant_village_train.py -l models/akidanet_plant_village_iq8_wq4_aq4.h5 -s models/akidanet_plant_village_qat.h5 -e 2 -lr 1e-4 $DATA_ARG

python plant_village_eval.py -l models/akidanet_plant_village_qat.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_plant_village_qat.h5
python plant_village_eval.py -l models/akidanet_plant_village_qat.fbz $DATA_ARG

python plant_village_benchmark.py -l models/akidanet_plant_village_qat.fbz $DATA_ARG

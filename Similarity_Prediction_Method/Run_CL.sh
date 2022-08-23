for dataset_name in flask pgcli keras
do
    python Prediction_by_CL.py --standard_dir ..\\data\\TLR\\standard\\$dataset_name --task_type source
    python Prediction_by_CL.py --standard_dir ..\\data\\TLR\\standard\\$dataset_name --task_type target
done
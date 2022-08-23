for dataset_name in flask pgcli keras
do
    python Prediction_by_VSM.py --standard_dir ..\\data\\TLR\\standard\\$dataset_name
done
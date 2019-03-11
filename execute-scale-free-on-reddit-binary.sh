python preprocess-dataset.py -n REDDIT-BINARY -k 10 -s
python preprocess-dataset.py -n REDDIT-BINARY -k 17 -s
python run-ego-cnn.py -n REDDIT-BINARY -k 17 -g 0 -f 0.99 -b 10 -m 6L
python run-ego-cnn.py -n REDDIT-BINARY -k 17 -g 0 -f 0.99 -b 10 -m 2L
python run-ego-cnn.py -n REDDIT-BINARY -k 17 -g 0 -f 0.99 -b 10 -m 6L_SF
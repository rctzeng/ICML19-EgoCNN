python preprocess-dataset.py -n MUTAG -k 10 -s
python preprocess-dataset.py -n MUTAG -k 17 -s
python run-ego-cnn.py -n MUTAG -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n PTC_MR -k 10 -s
python preprocess-dataset.py -n PTC_MR -k 17 -s
python run-ego-cnn.py -n PTC_MR -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n PROTEINS -k 10 -s
python preprocess-dataset.py -n PROTEINS -k 17 -s
python run-ego-cnn.py -n PROTEINS -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n NCI1 -k 10 -s
python preprocess-dataset.py -n NCI1 -k 17 -s
python run-ego-cnn.py -n NCI1 -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n DD -k 10 -s
python preprocess-dataset.py -n DD -k 17 -s
python run-ego-cnn.py -n DD -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n IMDB-BINARY -k 10 -s
python preprocess-dataset.py -n IMDB-BINARY -k 17 -s
python run-ego-cnn.py -n IMDB-BINARY -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n IMDB-MULTI -k 10 -s
python preprocess-dataset.py -n IMDB-MULTI -k 17 -s
python run-ego-cnn.py -n IMDB-MULTI -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n REDDIT-BINARY -k 10 -s
python preprocess-dataset.py -n REDDIT-BINARY -k 17 -s
python run-ego-cnn.py -n REDDIT-BINARY -k 17 -g 0 -f 0.99 -b 10

python preprocess-dataset.py -n COLLAB -k 10 -s
python preprocess-dataset.py -n COLLAB -k 17 -s
python run-ego-cnn.py -n COLLAB -k 17 -g 0 -f 0.99 -b 10
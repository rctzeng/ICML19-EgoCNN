import os
import optparse
import urllib.request
import zipfile

"""
Download Graph Classification datasets 
from [TU Dortmond](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
"""

DATASET_DIR = 'dataset'
DATASET_LIST = ['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'DD', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']
URL_BASE = 'https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets'

def parse_arg():
    parser = optparse.OptionParser()
    parser.add_option('-n', dest='all_or_specify_dataset', help='download "all" or specify the name of dataset', default='all')
    (options, args) = parser.parse_args()
    return options


def download_dataset(name):
    zip_path = '{}/{}.zip'.format(DATASET_DIR, name)
    urllib.request.urlretrieve('{}/{}.zip'.format(URL_BASE, name), zip_path)
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(DATASET_DIR)
    zip_ref.close()
    os.remove(zip_path)

def download(all_or_specify_dataset):
    # create directory to hold dataset if not exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    if all_or_specify_dataset == 'all':
        for name in DATASET_LIST:
            download_dataset(name)
    elif all_or_specify_dataset in DATASET_LIST:
        download_dataset(all_or_specify_dataset)

opt = parse_arg()
download(opt.all_or_specify_dataset)
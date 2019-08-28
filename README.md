# Few-shot-KGC

Requirements: python3, pytorch1.0

First, download two datasets from here: https://drive.google.com/open?id=1bekOAfMrx9V3uUp6dSYWkr-L5f3fTkwP, and put them into Few-shot-KGC/data/.

Train a model on Wikidata: python main_reptile.py --train --dataset wikidata --idx_device -1

Train a model on DBpedia: python main_reptile.py --train --dataset dbpedia --idx_device -1

Please note: --idx_device -1 means to train model on CPU. If you want to train it on GPU, replace -1 to your GPU index. For example, --idx_device 0. The code cannot simutaneously run on multiple GPUs.

Dataset-specific hyper-parameters are in dataset_config.py, and other hyper-parameters are in hyp_reptile.py.



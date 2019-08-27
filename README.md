# Few-shot-KGC

First, download datasets from here: https://drive.google.com/open?id=1bekOAfMrx9V3uUp6dSYWkr-L5f3fTkwP, and put data/ into Few-shot-KGC/.

Train a model on Wikidata: python main_reptile.py --train --dataset wikidata --idx_device -1

Train a model on DBpedia: python main_reptile.py --train --dataset dbpedia --idx_device -1

Please note: --idx_device -1 means to train model on CPU.

If you want to train it on GPU, replace -1 to your GPU index. For example, --idx_device 0.



# [Stop&Hop](https://arxiv.org/abs/2208.09795): Early Classification of Irregular Time Series ![Github_Picture](https://user-images.githubusercontent.com/26936677/187677997-5c230cef-af3c-4644-87f9-151675c30a7f.jpg)

This repository includes all necessary components to replicate our experiments and use our proposed method.

## Working examples

We provide two examples for using Stop&Hop:

### Pretrained Embeddings
First, we provide dataloaders from an RNN pretrained on the physionet dataset, which can be found in `pretrained_example.py`.
To run this example, you can create a virtual environment from our requirements.txt file:

```
python3 -m venv stophop_venv

source stophop_venv/bin/activate
```

### Joint Learning for RNN and HaltingPolicy
Second, we provide access to our ExtraSensory datasets, which can be run using `example.py`.

#### Citation
Please use the following to cite this work:
```
@inproceedings{hartvigsen2022stophop,
  title={Stop\&Hop: Early Classification of Irregular Time Series},
  author={Hartvigsen, Thomas and Gerych, Walter and Thadajarassiri, Jidapa and Kong, Xiangnan and Rundensteiner, Elke},
  booktitle={Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
  year={2022}
}
```
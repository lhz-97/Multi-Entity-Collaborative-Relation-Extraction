# Multi-Entity-Collaborative-Relation-Extraction

- ## Dataset

  We have done experiments on two data sets. ACE 2005 dataset needs License, and WIKIE has been uploaded to the dataset folder. You can run the following code to preprocess the dataset.

  ```
  python process_dataset.py
  ```

- ## Requirements and Installation

  This repository has been tested with `Python 3.6`,`torch==1.4.0`,`sacred==0.8.1`

  ```
  pip3 install -r requirements.txt
  ```

- ## Get Started

  ### Running

  **train**:

  The parameters of the function `main` are set in config.py. You can modify the parameters in it to train different models. And the hyper parameters for the models are also stored in it.

  After configure the file, you can run the following code for training.

```
bash runtrain.sh
```


# Multi-Entity-Collaborative-Relation-Extraction

- ## Dataset

  We have done experiments on two data sets. ACE05 dataset needs License, and WIKIE has been uploaded to the dataset folder. You can run the following code to preprocess the dataset.

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

  The parameters of the model are set in `config.py`. You can modify the parameters in it to train different models. And the hyper parameters for the models are also stored in it.

  After configure `config.py`, you can run the following code for training.

  ```
  bash runtrain.sh
  ```



## Additional experimental analysis

- ### Influence of GCNs Layers

<center> <img src="/figure/table1.png" alt="table1.png" width="60%" height="60%" /> </center>

We try different layers of GCNs. As shown in Table 1, for the ACE05 data set, one layer GCNs will be slightly better, which is related to the density of the     relationship in the sentence. For the WIKIE, due to the high relationship density, it is better to use the three-layer GCNs. This verifies the correctness of our model using GCNs to interact with entities and relationships.

- ### Influence of Different REs on NER

<center> <img src="/figure/table2.png" alt="table2.png" width="55%" height="55%" /> </center>

We  use  the  method  of  multi-task  learning  to  improve  their performance through the joint modeling of entities and relations. As shown in Table 2, You can see that the performance of NER has been improved with the use of different RE models.  In general, with the improvement of relation extraction, the performance of NER is also improving.

- ### Case Study
<center> <img src="/figure/table3.png" alt="table2.png" width="90%" height="90%" /> </center>

Table 3 shows qualitative results that compare our model with baseline models.  For S1, through the “locatedin” relation between “council ward” and “Lower Clapton” and the “locatedin” relation between “Lower Clapton” and “London Borough of Hackney”, our model infers that the relation between “council ward” and “London Borough of Hackney” is “locatedin”.  For S2, the relation between “Richaun Diante Holmes” and “Philadelphia 76ers” is “playerof”, and the relation of “Richaun Diante Holmes” and “basketball” is “profession”.  Our model reason that the relation between “Philadelphia 76ers” and “basketball” is “domain”.  Of course, it may be inferred through the “National Basketball Association” for the middle word.  For S3, the relation between “Kung Fu Panda” and “Jack Black” is “actor”.  The relation of “sequel” and “Jack Black” is “actor”. Our model concludes that the relation between “sequel”and “Kung Fu Panda” is “partwhole”.  Obviously, compared with other models, our model performs better in multi-pair relation extraction tasks.

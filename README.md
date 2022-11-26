# CMGCN
[ACL 2022] The source code of Multi-Modal Sarcasm Detection via Cross-Modal Graph Convolutional Network
This is the code of the proposed CMGCN model for multi-modal sarcasm detection

*FILES*
    1 ./train.py : code for running the training and testing procedures
    2 ./bucket_iterator.py : code for converting the inputs into pytorch tensors
    3 ./data_utils.py : code for loading the dataset and performing BERT tokenization
    4 ./glove_embedding.py : code for loading the dataset and performing GloVe tokenization
    5 ./generate_cross_modal_graph.py : code for generating the cross-modal graph
    6 ./layers/*.py : code for some modules used in our model
    7 models/CMGCN.py : code for our model
    8 ./get_boxes.ipynb : code for get boxes and attribute-object pairs of the images
    9 ./get_VITfeats.ipynb : code for get the ViT features of the boxes

*SET UP*
    1 You can get the BERT-base pretrained language model from "https://huggingface.co/bert-base-uncased" and put it in ./bert_base_uncased
    2 You can get the ViT pretrained model we used from "https://github.com/lukemelas/PyTorch-Pretrained-ViT"
    3 You can get the image caption model we used from "https://github.com/peteanderson80/bottom-up-attention"
    4 You can get the multi-modal sarcasm detection dataset from "https://github.com/headacheboy/data-of-multimodal-sarcasm-detection"
    5 The *.ipynb files and ./generate_cross_modal_graph.py can help you to perform preprocessing of the data, you may follow the comments in them.
    6 After all the above done, you may run the run.sh file to train the model.

*HYPER-PARAMETERS*
    --model_name : name of the model to run
    --dataset : name of the processed dataset to use
    --optimizer : name of optimizer to use 
    --initializer : name of initializer to use
    --lr : learning rate
    --dropout : dropout rate in the training procedure
    --l2reg : L2 regularization
    --num_epoch : number of epoch to run
    --batch_size
    --log_step : number of step for logging
    --patience : patience of epoch for early stop
    --device : GPU to use
    --seed : random seed
    --macro_f1 : whether to use Macro-F1
    --pre : whether to perform predicting
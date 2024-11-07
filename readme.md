# Simple Transformer

## Introduction
This is my implementation of the original transformer paper, this was basically a learning experience for me.

I tried to make the implementation as easy to follow as possible, with each section of the transformer having

its own class. I will also be updating it the future adding perhaps more clarity to certain parts in the from 

of variable renaming, maybe some slight refactoring and also some additional comments, but for the most part

this is it.

## Running
- Generally to run this, you can just clone the directory into whatever folder you want and run it either from within
  
    a code editor or from command prompt, theres certain aspects of the transformer/dataset that you can change but these

    are all self explainatory in the train.py file and translate.py file and can be changed depending on what you want to do. I did include

    two "pretrained" files that translate from english to french and french to english that you can play around if you don't

    want to fully train the model from scratch.

- I trained this on a GTX 3060 which has 12GB of VRAM so if you want to train the model yourself you might need to adjust the token size, if
  
    you have less lower the token amount or if you have more increase the token amount etc.

- Each Epoch took around 7-8 minutes, so for 20 or so epochs you're looking at around 2-3 hours to train the entire model on the iwslt dataset.
  
  If you're planning to train on the wmt dataset on a single gpu.......good luck.

## Notes
- **Pre-Norm instead of Post-Norm**:
  
    This implementation uses a ***PRE-NORM*** paradigm for the transformer, instead of the ***POST-NORM*** used in the original paper,

    this is mostly due to the fact that when you use a low amount of data convergence becomes very difficult/impossible without

    switching to a pre-norm paradigm, for more details theres a really good paper which explains this better than I can here called

    [Transformers Without Tears](https://arxiv.org/abs/1910.05895).

- **Dataset Used**:
  In this implementation I used the 2017 IWSLT dataset from english to french and used the validation set to check the performance,

    this provided a good balance between quick iterations in terms of training speed but also having enough size that a transformer of this size could

    be meaningfully trained on it.

## Results
 |IWSLT 2017 Train|IWSLT Val|
 |-----|-----|
 |EN-FR|23.00 Bleu
 |FR-FR|21.02 Bleu

## Acknowledgements

A huge help in implementing partial aspects like the positional encodings and the label smoothing came from [Aleksa Gordić](https://github.com/gordicaleksa/
    pytorch-original-transformer) 

and his implementation and were instrumental in getting the final product ready to train.

Also the countless youtube videos that helped understanding on nauance aspects of transformers

that go beyond thr scope of just the original transformer and in to its variants like the reformer, Bert, GPT,Linformer etc.

## Blog
I wrote a blog on this project which goes into much more detail on some of the challenges of training this model, you can find it [here](https://medium.com/@christopherdoherty14/how-not-to-train-your-transformer-7e63011a16eb)

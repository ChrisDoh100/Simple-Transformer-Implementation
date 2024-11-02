# Simple Transformer

## Introduction
This is my implementation of the original transformer paper, this was basically a learning experience for me.
I tried to make the implementation as easy to follow as possible, with each section of the transformer having
its own class. I will also be updating it the future adding perhaps more clarity to certain parts in the from 
of variable renaming, maybe some slight refactoring and also some additional comments, but for the most part
this is it.
## Running
Generally to run this, you can just clone the directory into whatever folder you want and run it either from within
a code editor or from command prompt, theres certain aspects of the transformer/dataset that you can change but these
are all self explainatory in the train.py file and translate.py file and can be changed depending on what you want to do. I did include
two "pretrained" files that translate from english to french and french to english that you can play around if you don't
want to fully train the model from scratch.
## Notes
- **Pre-Norm instead of Post-Norm**:This implementation uses a ***PRE-NORM*** paradigm for the transformer, instead of the ***POST-NORM*** used in the original paper,
    this is mostly due to the fact that when you use a low amount of data convergence becomes very difficult/impossible without
    switching to a pre-norm paradigm, for more details theres a really good paper which explains this better than I can here called
    [Transformers Without Tears](https://arxiv.org/abs/1910.05895).
- **Dataset Used**: In this implementation I used the 2017 IWSLT dataset from english to french and used the validation set to check the performance,
    this provided a good balance between quick iterations in terms of training speed but also having enough size that a transformer of this size could
    be meaningfully trained on it.

## Results
 | Dataset | Eval |
 |IWSLT 2017 Train|IWSLT Val|
 |-----|-----|
 |EN-FR|23.00 Bleu
 |FR-FR|21.02 Bleu

## Acknowledgements
A huge help in implementing partial aspects like the positional encodings and the label smoothing came from [Aleksa Gordić](https://github.com/gordicaleksa/pytorch-original-transformer) and his implementation and were instrumental in getting the final product ready to train.

## Blog
I do plan on writing a blog on this project which will go much more in depth on some of the pitfalls of creating this project, I'll link it here when its ready.
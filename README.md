## Text Generation with Feed Forward Neural Networks 

Text Generation can be be modeled using deep learning models such as Feed-Forward
Neural Networks (FFNN) and Recurrent Neural Networks (RNN). For the text
generation task, FFNNs are trained on a very large corpus to predict the next word as
a bigram language model. Once the model is trained, it is straightforward to generate a
new text by iteratively predicting the next word as a n-gram language model .
I implemented an n-gram (bigram) level FNN for Text Generation 
by using DyNet1 deep learning library.

The idea behind FNN language model with n-gram is that words in a word sequence
statistically depend on the words closer to them, only n-1 direct predecessor words are
considered when evaluating the conditional probability.

After build FNN language model, trained it to use for poem generation.
To generate a new poem, it needed to start with start token. Then, predict one
word at each time using the previous word and feeding this word to the input of the
next time.


Dataset
A poem dataset called Uni-Modal Poem, which is a large poem corpus
[dataset](https://drive.google.com/file/d/11GLY2J_B106F_aapZfRmOlhcn1vhQCHG/view) that involves around 93K poems. UniM-Poem is crawled from several publicly
online poetry web-sites, such as Poetry Foundation, PoetrySoup, best-poem.net andpo-
ets.org. For the pretrained word vectors, GloVe 6B word embeddings used.

# textsnippet
A. NLTK seq 2 seq
This model uses nltk to implement stop words, stemmer,
tokenizer modules. Initially the raw test data is passed to this
model. All the contractions are expanded first like aren’t”:
”are not”, can’t”: ”cannot”, could’ve”: ”could have, he’s”: ”he
has / he is” etc. First the whole paragraph is converted into
sentences using nltk Punkt sentence tokenizer. All the stop
words, special characters, and numbers from all the sentences
are removed using nltk stopwords. Then all the words into
sentences are tokenized. The frequency table is created using
these tokens of word which represent the weights frequency of
word occurrences. This weight frequency is replacing the word
in original sentence. Then the sentence value is calculated,
and the sentences are sorted into descending order with their
sentence value. The sentences with the highest frequencies are
used to generate summary.

B. LSTM
This model uses LSTM encoder and decoder to generate
summary. Before that text preprocessing is done by removing
white spaces, contraction expansion, converting all text to
lower and removing special characters. Then dataset is divided
into train and test data here we have divided our data into
70% train and 30% for test. These data are tokenized by word
tokenizer and send it to encoder. But before that we add the
embedding layer which has low dimension space so we can
translate the data into high-dimensional vectors. It will help
in building models on the large input data.
We have implemented 3-layer LSTM encoder in stack manner
which helps us to increase the model complexity which
leads to better representation of input sequences. Encoder
reads the input data at each time step; one word is forward to
the encoder and processes the information at every timestep.
Which capture the contextual data available into given input
data sequence. LSTM encoder layer uses the return state and
return sequence which produces hidden state and cell state
at each timestep and last timestep respectively. We can also
provide number of hidden states as latent dim value.
Now the output of the encoder with hidden state and cell
state is fed as the LSTM decoder. Hidden and cell state of
last timestep is used in initializing the decoder initial state.
While decoder reads the entire target sequence word by word
and predict the same offset at each timestep. As the decoder is
trained to predict the next word in the sequence by the word
provided in last given output.
Attention Mechanism worked as the production of the next
word in the summary by guiding the decoder where to search
in the target words. This process created the weighted number
of the encoder’s hidden state, which aided in the generation
of the context vector, which is the fixed size representation of
the input.

C. Bidirectional LSTM
This model is like LSTM model, but it trains two layers
instead of one layer in LSTM for input sequence. Two layers
are forward layer and backward layer: The forward layer
calculates the hidden and cell states similar as standard LSTM
encoder. The backward layer calculates them by taking the
input sequence in a reverse order by starting from last time
step to reaching first timestep of input sequence. The main
idea of using a backward layer is that we can create a path
which provides vision for predicting future data and helps in
learning its weights. This probably helps the node to capture
the dependencies which may not captured by LSTM. For
encoding, the hidden and cell states of the forward layer is
concatenated with states of the backward layer respectively.
The main idea of bidirectional LSTM decoder is to address
the problem of unbalanced summary. A summary imbalance
occurred due to noise in a last prediction that leads to reduction
in quality of generate summaries. The forward decoder and the
backward decoder are the two LSTMs layers that used by the
bidirectional decoder. The forward decoder layer decodes the
information from left to right, and the backward decoder layer
decodes the information in reverse order compare to forward
layer. The last hidden state of the forward decoder layer is
used as the initial input to the backward decoder layer. The
decoder output is depending on the context vector of encoder
and the hidden state of decoder. The input of forward decoder
layer is the reference for the last summary token.
The decoder performance is determined by the encoder’s
internal representation, which includes the context vector,
the decoder’s current hidden state, and the summary terms
previously provided by the hidden states. The purpose of
training is to maximise the chance of consistency between
the statement and the summary in all ways as far as possible.
The previous comparison description token is used as the
forward decoder’s feedback during preparation. During
testing, however, the token created in the previous phase is
used as the forward decoder’s input. The backward decoder
is in a similar position, with the potential token from the
description as the input during testing. In other words, it
would give the network more meaning and, as a result, it will
be faster than LSTM.

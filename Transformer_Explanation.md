# Transformers

# Table of Contents
- [Conceptual Explanation](#conceptual-explanation)
- [Tokenization and Embedding](#tokenization-and-embedding)
- [Attention](#attention)
- [A Forward Pass through a Transformer](#a-forward-pass-through-a-transformer)
- [Preface to Code](#preface-to-code)
- [Sources](#sources)

## Conceptual Explanation

The following assumes you have some knowledge of:

- Calculus  
- Derivatives in context  
- Matrix-vector products  
- A basic idea of machine learning  
- Multi-Layer Perceptrons  
- PyTorch tensors and their operations  

Transformers are the hallmark of modern Artificial Intelligence. They are a neural network architecture that specializes in processing sequences of information, such as sounds, text, and other forms of signals. They form the backbone of modern LLMs, including ChatGPT (Generative Pretrained Transformer), DeepSeek, Gemini, Google’s AI assistant, and more. How these well-established models work under the hood will be explained here. This document specifically addresses the Transformer architecture which is known for text generation, or “prompting.”

Here are some important terms to look for as you read. If you have a good understanding of what these mean, then you are already at a decent understanding of Transformers:

- Tokens  
- Embeddings  
- Keys  
- Queries  
- Values  
- Attention  
- Masking  

Before continuing, it's best to get a sense of what a Transformer really is. Like other machine learning models, Transformers are collections of learnable parameters and activation functions that are structured in a specific way to address and optimize a task. In the specific case of a Transformer, the task is to be able to predict what comes next in a sentence.

For example, a Transformer can be seen as a trained and much more sophisticated autocomplete, where based on the previous words in a sentence, the one that comes next is predicted. But in order for someone to know what comes directly next in a sentence, they have to know the meaning of words in a sentence, the context of how each word is used, and what generally follows certain words.

How does a Transformer do this?


### Tokenization and Embedding


When I said that Transformers predict the next word in a sentence based on all of the previous words, that might have been slightly inaccurate. Transformer models process what’s called “tokens.” Tokens are items created through a process of encoding a word or character string as a number or list of numbers and storing it in a vocabulary. Most Transformers don’t store words as words, but instead as chunks of them.

Of course, storing entire words is doable, but there are advantages and heavy disadvantages to doing so. The most commonly used method (used by models like ChatGPT, Gemini, etc.) is Byte Pair Encoding (BPE). BPE tokenization is the process of storing a “vocabulary” of the most common pairs or triplets of characters that appear in a certain text. This can be done in many different ways—through manual counting and merging or using prebuilt tokenizers where this process is already done (or even optimized).

One example of this is the sequence “th” becoming a token in a model’s vocabulary if words like “the,” “that,” or “there” are extremely common. For now, thinking in words is the most intuitive way to understand what’s to come, but it's worth knowing how most tokenization is actually done.



For a Transformer to be able to make predictions based on tokens, it has to have a way of representing them so that later context can be added to said representation. Because of this, Transformers store what are called “embeddings.” Embeddings are vectors that represent tokens in a vocabulary, and these are the main point of interest in a Transformer.

Upon initialization and vocabulary definition, Transformers store a lookup table that links every single token to an embedding (a matrix that links columns to tokens). These embeddings can be treated as representations of words without any context, which will later be updated when a Transformer processes a sentence. These are also learnable, which means that over time a Transformer performs backpropagation and updates the embedding table so that it can form better context-free representations.

These embedding vectors are often extremely high-dimensional (often vectors of 512 to 768 dimensions). You can consider this as giving space for Transformers to learn several “dimensions” of information about a token. The field of Computational Linguistics, which is how to program the process of understanding the meaning of words, is interesting in its own right, but we won’t delve too deeply here.

One fascinating example of how this could be was brought up by 3Blue1Brown (if you aren’t familiar with him, he’s a YouTuber who makes fascinating videos on computing, math, and AI). He explains how models that store these types of embeddings might have the difference between the vectors representing “man” and “woman” similar to the difference between the vectors representing “king” and “queen.”

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/f/fe/Word_embedding_illustration.svg" 
       alt="Word Embedding Illustration (World Capitals)" 
       width="700"/>
</p>
<p align="center">
  <em>“Word embedding illustration” by Fschwarzentruber (2025), licensed under CC BY-SA 4.0 via Wikimedia Commons.</em>
</p>


Now that you have an intuition for how embeddings work, we’ll get into the bulk of how Transformers work.



### Attention

When a sentence is passed into a Transformer, it will be tokenized, then stored as a list of embedding vectors.

$$
\begin{array}{ccccc}
\text{The} & \text{man} & \text{walked} & \text{the} & \text{dog} \\
\begin{bmatrix}
e_{11} \\ e_{21} \\ e_{31} \\ \vdots \\ e_{d1}
\end{bmatrix} &
\begin{bmatrix}
e_{12} \\ e_{22} \\ e_{32} \\ \vdots \\ e_{d2}
\end{bmatrix} &
\begin{bmatrix}
e_{13} \\ e_{23} \\ e_{33} \\ \vdots \\ e_{d3}
\end{bmatrix} &
\begin{bmatrix}
e_{14} \\ e_{24} \\ e_{34} \\ \vdots \\ e_{d4}
\end{bmatrix} &
\begin{bmatrix}
e_{15} \\ e_{25} \\ e_{35} \\ \vdots \\ e_{d5}
\end{bmatrix}
\end{array}
$$






Before processing the sentence, the model will add what’s called positional encoding vectors to each vector in the sentence. This vector can be created in many different ways (through learning or with functions), but either way, the purpose of it is to add values to each vector that signify its position in a sentence. This is a small step, but it makes a large difference in helping a machine learning model understand the positions of every word.

Now, the list of embeddings is passed through “attention heads.” Attention heads are components of a Transformer that help add context to each token.

The first thing to understand before getting into attention is that attention heads work in a process that involves three different features:

- **Queries** - The context that a token requires  
- **Keys** - The context that a token could give to other tokens  
- **Values** - The value of added context to a token  

These vectors—called the query vectors, key vectors, and value vectors—are created from matrix-vector products of three different kinds of matrices applied to each embedding: the query matrix, key matrix, and value matrix.

$$
\begin{array}{c}
\text{Embedding Vector } x \\\\[6pt]
\downarrow \\\\[6pt]
\begin{array}{ccc}
Q = xW^Q & K = xW^K & V = xW^V
\end{array}
\end{array}
$$

These matrices are learnable over time, as the process of training any kind of deep learning model on this task with a specific enough structure will lead the model to necessarily conform to it. While we don't know with certainty that the query, key, and value vectors serve the function that we believe them to, we can likely infer it due to the structure that we gave the model.

The second important thing to understand before you proceed into attention is an intuitive sense for what dot products mean. (If you are not familiar with linear algebra, please look up dot products and matrix-vector products before getting into what follows.)

The dot product involves multiplying the components of each vector at the same positions and summing these products:

$$
\text{Vector A} \cdot \text{Vector B} 
= \sum_{i=1}^{n} A_i \cdot B_i
= A_1 B_1 + A_2 B_2 + A_3 B_3 + \dots + A_n B_n
$$


Dot products represent the magnitude of similarity between two vectors. The higher a dot product is (especially when all values in a vector are scaled within a certain range), the more “similar” two vectors are in direction. For embedding vectors in particular, it would be multiplying the aforementioned embedding dimensions together and adding up the products of each.

For a simplified example, the dot product between the embedding vector that represents a king and the embedding vector that represents a queen will involve multiplying all their corresponding dimensions together and adding all of these products up.

A matrix product, after all, is just a collection of the dot products of the column vectors of one matrix with the row vectors of another.

![Self-Attention Matrix Calculation (Part 2)](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)  
*Image from Jay Alammar, “The Illustrated Transformer” (2018), licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.*  

You might have already seen this coming, but the main feature of self-attention is **scaled dot product attention**, where the matrix of query vectors (not the same as the query matrix) created from the embeddings is multiplied by the matrix of key vectors transposed. This produces a new matrix, which contains the products of all the query vectors and all the key vectors. This new matrix is then multiplied with the value vectors to produce a set of new vectors, which you can think of as vectors that represent additional context that can be added to the original embeddings.

In this case, the key vectors can almost be considered as “answering the questions” of the query vectors, with each dot product representing the similarity between the context each embedding vector seeks (the query vectors) and the context given by each vector to other vectors in the sentence (the key vectors). After this, the key-query product is scaled down, usually by dividing each vector by the square root of the dimensionality of each vector. If the context vector was of size 1x64, then it would be divided by 8. Then the value vectors are multiplied with this product to produce updated context vectors.

Now, Transformers work with more than one self-attention head at a time. They use what’s called **multi-headed attention**, where the embeddings are transformed into lower-dimensional chunks used in each attention head. Then, the output context vectors that are produced from each attention head are stacked together, then passed through a linear layer (multiplying the list of context vectors by a matrix of weights and adding them to a bias vector), then added back to the original embeddings.

This “split” isn’t actually done outside of each attention head and is instead put into the part where the query, key, and value matrices are multiplied to each embedding vector. These matrices are of dimension `embedding_size x (embedding_size / number_of_attention_heads)`, which makes the key, query, and value vectors of dimension `1 x (embedding_size / number_of_attention_heads)`. This is mainly done so that the Transformer can have different heads that can capture different relationships within a sentence.

For an intuitive example, one head might be good at determining what the subjects and objects of each sentence are, while another might be good at identifying which adjectives describe which nouns, and so on.

![Multi-Head Self-Attention Recap](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)  
*Image from Jay Alammar, “The Illustrated Transformer” (2018), licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.*  

One thing to note before we continue is that oftentimes a **mask** is applied to the key × query product of each matrix. This **mask** is an upper triangular matrix with each value being -infinity. The main reason this is done is to “hide” the context that future tokens give to a token that comes before it and to make sure predictions are done left to right in a sentence. This can allow you to train your model much faster, as one training sentence can serve as several mini training examples.

$$
\text{Mask Matrix (Causal Mask)} \quad
\begin{bmatrix}
\text{token 1} & 0 & -\infty & -\infty & -\infty \\
\text{token 2} & 0 & 0 & -\infty & -\infty \\
\text{token 3} & 0 & 0 & 0 & -\infty \\
\text{token 4} & 0 & 0 & 0 & 0
\end{bmatrix}
$$


There’s much more nuance to why this is done, but it’s all you need to think of when considering masks for now.

For now, consider the entire process that occurs (except for the tokens being turned into embeddings) as part of a **multi-headed attention block**, done with several attention heads at once.


### A Forward Pass through a Transformer

Like most other machine learning models, a Transformer is split into layers. These layers are usually composed of multi-layer perceptrons, multi-headed attention blocks, layer Norms, and layer dropout. You should know the first two by now, but to go over the latter:

- **Layer norms** are functions applied to vectors of data that reduce the variance between dimensions in a vector by subtracting the mean of all of them and dividing by the standard deviation. For an MLP, this would be applied to the activation vector that each layer produces, and for a Transformer this would be applied to the dimensions of the updated embeddings (the sum of the original embeddings and the context vectors).

- **Layer dropout** is a function that randomly disables parts of a model. This helps prevent the model from overfitting (when a model memorizes a dataset with extremely precise weights that fail to generalize).

In each Transformer layer, the input embeddings pass through attention and then through two linear MLP-style layers (all wrapped with dropout blocks). Two layer norms are usually applied, either before attention and before the MLP pass or after attention and after the MLP pass. There’s nuance to how this can be done (especially with how the MLP layers are used), but you can read about it later.


#### Multi-Headed Attention Diagram

![Multi-Head Attention Diagram](https://upload.wikimedia.org/wikipedia/commons/6/68/Attention_Is_All_You_Need_-_Multiheaded_Attention.png)  
*Image from “Attention Is All You Need” (Vaswani et al., 2017), licensed under CC BY-SA 4.0, used for educational purposes.*

**Attention Head**  

[Embeddings] -> [Key Vectors, Query Vectors, Value Vectors] -> [Scaled Dot Product] -> [Masking] -> [Key-Query product x Value Vectors]

**Multi-Headed Attention Block**  

[Embeddings] -> [Layer Norm] -> [Multi-Headed Attention] -> [Layer Dropout] -> [Adding Input Embeddings] + [Layer Norm] -> [MLP Layers] -> [Layer Dropout] -> [Adding Input Embeddings]


**Entire Transformer Forward Pass**  

[Sentence] -> [Tokenization] -> [Embeddings] -> [Positional Encoding] -> [Transformer Layers] -> [Layer Norm] -> [Linear Projection] -> [Softmax] -> [Outputs]


Once the tokens of a sentence are turned into embeddings, positional encoding is added, and the embeddings are fed through all the Transformer layers, we have embeddings that hopefully capture all the context within a sentence, with each embedding having context from the embeddings that came before it.

We then apply another layer norm to these embeddings before performing a matrix-vector product, where a matrix of size `vocab_size × embedding_size` is multiplied by the list of embeddings, and a bias vector is added. This matrix is usually the same as the embedding lookup table, which is a choice that usually leads to better performance within a Transformer (look up “weight tying”).

When you train a Transformer, you would typically feed this result to **Categorical Cross-Entropy**, the loss function specialized for models that need to predict across distinct classes (in this case, tokens). The function outputs a loss value representing how off the model was regarding the probabilities of each word being next.

For predicting the next token in a sentence, a softmax function is applied to the final list, producing a probability distribution that represents the likelihood of a certain token being the next one in a sentence. One simple way to generate new tokens is to pick the token with the highest probability and output it. While intuitive, this is not what typical Transformers use. Instead, they often sample from the “top k” highest probability words, which works better for text generation and helps avoid repetitive text.

That wraps up the process of how a Transformer predicts the next token in a sentence based on all previous tokens. The mechanisms described earlier are the backbone of how large language models like ChatGPT, DeepSeek, Gemini, and many others work. Note that these models are far larger than a standard Transformer and are trained in unique ways beyond simple sentence feeding and token prediction. Many details are proprietary, but there is open information worth reading about.



### Preface to Code

For efficient training, Transformers must be trained in a **vectorized style**, which allows processing multiple batches of data simultaneously. Without vectorization, training is almost impossible.

Transformers also generally require GPUs to run and train properly. Students at universities can access Google Colab’s Pro GPUs for free. If you plan to run this code, it is strongly recommended to use these GPUs, as the free T4 provided by Colab may require scaling down the model and dataset significantly.

Instead of manual backpropagation, this Transformer uses PyTorch’s **Adam Optimizer**, popular in ML optimization. Understanding Adam provides insight into formal optimization methods under the hood.

The code walks through all the classes involved in Transformers, the training process, and shows validation and prompting in Colab. It is strongly recommended to read the code thoroughly to understand the logic behind the implementation.

Another note is that the code I have doesn't re-define vocab_size when loading in a prebuilt transformer. This is to give some level of flexibility as to whether you wish to train your transformer from a checkpoint or evaluate with the generate method. If you don't want to go through the process of reloading the corpus, put this statement in the block "Loading a saved model":

vocab_size = 17500 #Or whatever your old vocab size is

#### Saved Results

https://drive.google.com/drive/folders/1ErROb5dYt3T5LoL_FGpAhMo0ZA3yXNBv

In this folder, parameters for a model capable of intelligible phrase completion are provided. The model can generally complete sentence fragments and short additions but is not large enough to capture dependencies longer than one or two sentences. These parameters can be loaded after rebuilding the vocabulary and dataset. GitHub has issues when rendering notebooks with output results(Example_Transformer_Uploaded_Results), so if you want to play around with the model's saved params I recommend downloading the notebook ipynb and uploading it to a colab/jupyter environment, then downloading the saved tokenizer and transformer parameters and uploading them to your preferred execution environment.



## Sources

- 3Blue1Brown. *But what is a Neural Network?* YouTube, uploaded by 3Blue1Brown.  
  https://www.youtube.com/watch?v=wjZofJX0v4&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6  

- Alammar, Jay. *The Illustrated Transformer.*  
  https://jalammar.github.io/illustrated-transformer/  

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).  
  *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.  
  https://arxiv.org/abs/1706.03762  

- Hugging Face. *The LLM Course: Attention and Transformers.*  
  https://huggingface.co/learn/llm-course/chapter1/4?fw=pt  







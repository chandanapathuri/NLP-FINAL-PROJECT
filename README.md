# NLP-FINAL-PROJECT

 # Sequence to Sequence Learning with Neural Networks

## Introduction

A Question Answering system is an AI-based system designed to offer coherent responses to user queries. It examines user questions through the application of natural language processing, machine learning, and other advanced technologies. Subsequently, it retrieves information from either a database or the internet and presents the solution in a manner comprehensible to humans. These systems serve various purposes by minimizing the time and effort required for information retrieval. In the contemporary world, where access to extensive information is prevalent, question-and-answer systems have become essential tools for facilitating efficient communication.

## Methodology

### Overview

The operational framework of this question-and-answer system is intricately designed to cater to the diverse needs of users, providing an effective mechanism for extracting pertinent information from a given context. The systematic approach encompasses a series of well-defined steps, ensuring a comprehensive and accurate response to user queries. The process unfolds as follows:

Commencing the inquiry journey, the system incorporates a sophisticated question processing component. This component meticulously reads and discerns keywords embedded within user queries, laying the groundwork for the subsequent stages of information retrieval.

Upon receipt of the user's textual input, the system seamlessly transitions into a corpus processing component. This integral phase involves an in-depth exploration of the context, scouring for relevant information that corresponds to the identified keywords. Leveraging advanced techniques, this step aims to enhance the precision and efficiency of information extraction.

Culminating in the final stages of the methodology, the system meticulously retrieves the output. This retrieval process is finely tuned to ensure that the user's mandatory query is not only addressed but also presented in a coherent and understandable manner. The system thus serves as a sophisticated intermediary, bridging the gap between user inquiries and the vast reservoir of contextual information.

In essence, this methodical approach underscores the system's commitment to optimizing the user experience by streamlining the complexities of information retrieval, thereby contributing to the seamless exchange of knowledge in the contemporary digital landscape.

## Approach

### Phases

1. **Web Scraping the Data:** Initiating the project involves extracting data from the web. This crucial step sets the foundation for subsequent processes by collecting pertinent information.

2. **Creating Custom Dataset and Dataloader:** With the acquired data, we move on to constructing a custom dataset and developing a dataloader. This step ensures efficient handling and organization of the data for subsequent model training.

3. **Token Generation using Regular Expressions:** Employing regular expressions, we tokenize the data, breaking it down into meaningful units. This step is pivotal in preparing the data for further processing.

4. **Generating Vocabulary:** Building on the tokenized data, we generate a vocabulary that encapsulates the diverse elements present in the dataset. This serves as a foundational resource for subsequent language-based tasks.

5. **Generating Dictionary:** In tandem with vocabulary generation, we create a dictionary to systematically map tokens to numerical representations. This facilitates the integration of data into numerical models.
   
6. **Creating a Mini Network from Scratch:** The project involves the development of a mini network from scratch, tailored to the specific requirements of our objectives. This network serves as the backbone for subsequent model training.
   
7. **Defining Hyperparameters:** To optimize the model's performance, we meticulously define hyperparameters, fine-tuning the configuration to achieve the desired results.
  
8. **Training the Model:** The model undergoes a comprehensive training phase, where it learns to discern patterns and relationships within the dataset. This step is crucial for the model's efficacy in real-world applications.
    
9. **Model Evaluation:** Post-training, the model undergoes rigorous evaluation to gauge its performance and identify areas for improvement. This iterative process ensures the refinement of the model's capabilities.

10. **Transfer Learning & Hugging Face:** Leveraging the power of transfer learning, we integrate Hugging Face, a state-of-the-art natural language processing library. This step enhances the model's capabilities through the utilization of pre-trained models and resources. 


## Algorithms:

### Depthwise Separable Convolution

The depthwise separable convolution algorithm presents a distinctive approach to convolutional operations, markedly different from regular convolutions. Its primary advantage lies in its efficiency, achieved by reducing the number of required multiplication operations. This is accomplished by breaking down the convolution process into two distinct stages: depthwise convolution and pointwise convolution.

In contrast to conventional Convolutional Neural Networks (CNNs), where convolution is applied simultaneously across all M channels, the depthwise separable convolution executes convolution on individual channels sequentially. Specifically, the filters or kernels utilized in this process are dimensioned as Dk x Dk x 1. Given an input data with M channels, M such filters are employed. Consequently, the output of this operation results in a size of Dp x Dp x M.

To elaborate further, the algorithm follows these steps:

#### Depthwise Convolution:

- Application of convolution individually to each channel.
- Utilization of filters/kernels sized Dk x Dk x 1, where Dk represents the kernel dimensions.

#### Pointwise Convolution:

- Application of a 1x1 convolution across all channels simultaneously.
- This step aids in combining information from individual channels.

The culmination of these steps leads to an output tensor with dimensions Dp x Dp x M. By virtue of its sequential and specialized approach, the depthwise separable convolution algorithm demonstrates improved computational efficiency, making it particularly advantageous in scenarios where computational resources are a critical consideration.

### Highway Networks

Highway networks emerged as a solution to simplify the training of deep neural networks. Despite advancements in enhancing shallow neural networks, training deep networks posed challenges such as vanishing gradients. The structure described so far aligns with a common pattern found in Natural Language Processing (NLP) systems. While the advent of big pretrained language models and transformers has somewhat rendered this pattern obsolete, many NLP systems employed it before the era of transformers. The rationale behind this lies in the effectiveness of incorporating highway layers, allowing the network to leverage character embeddings more efficiently. This proves particularly useful when dealing with out-of-vocabulary (OOV) words, as a word can be initialized even if it is not part of the pre-trained word vector vocabulary.

### Embedding Layer

The embedding layer plays a pivotal role in the architecture, encompassing the following functionalities:

- Conversion of word-level tokens into a 300-dimensional pre-trained GloVe embedding vector.
- Generation of trainable character embeddings using 2-D convolutions.
- Concatenation of character and word embeddings, followed by passage through a highway network.

### Multi-Headed Self Attention

The cornerstone of the multi-headed self-attention mechanism involves computing the similarity between two representations, transforming them into an attention distribution, and subsequently aggregating values in a weighted manner. While the fundamental principle of attention remains consistent, specific nuances need to be addressed for optimal implementation.

### Positional Embedding

The model lacks inherent knowledge of word sequence in a phrase, a role traditionally handled by RNNs or LSTMs. To address this, positional embedding is introduced.

### Encoder Block

This layer involves injecting positional embeddings, passing through convolutional layers (four for the embedding encoder layer, two for the model encoder layer), and subsequent steps like a feedforward network and multi-headed self-attention. Residual connections ensure consistency.

### Context-Query Attention Layer

This layer bifurcates attention, revealing relevant query terms for each context word.

### Output Layer

The output layer predicts start and end indices of the answer within the context.

### QANet

This model integrates word-level and character-level tokens, processes them through embedding and encoder layers (with convolution and attention), and concludes with the output layer determining response indices.


## Installation and Usage

Refer the uploaded Report file.

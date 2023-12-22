In this assignment, we will learn and practice language models. As mentioned in the textbook, language modeling was defined as the task of predicting the next word in a sequence given the previous words. Therefore, we will focus on the related problem of predicting the next character in a sequence given the previous characters.

The learning goals of this assignment are to:

Understand how to compute language model probabilities using maximum likelihood estimation.
Implement basic smoothing and interpolation.
Understand how to evaluate alternative language models using perplexity.
Have fun using a language model to probabilistically generate texts.
Dataset
For Parts 0-2: Everyone

Skeleton python codeDownload Skeleton python code
training dataLinks to an external site. of Shakespeare
test data of New York Times article, and several of Shakespeare’s sonnets
For Part 3: Masters (2071) part of the assignment

Wikipedia ArticlesDownload Wikipedia Articles
Caesar Shifter Script  Download Caesar Shifter Scriptto aid in encrypting and decrypting messages
Encrypted Message  Download Encrypted Messagefor testing your approach
Deliverables
A writeup doc: writeup.pdf
At various points in this assignment you will be asked a questions to respond to or asked to Report various things.
A file of your updated skeleton python code: ngram_skeleton.py
A README.txt file explaining
how to run your code
the computing environment you used; what programming language you used and the major and minor version of that language
any additional resources, references, or web pages you've consulted
any person with whom you've discussed the assignment and describe the nature of your discussions
any unresolved issues or problems
something nice is letting me know what section you are in. Undergraduate vs Masters/1671 vs 2071
Grading (Undergraduate/1671)
Code (65 points):

5x2 points for correctly implementing 2 functions in Part 0
5x7 points for correctly implementing 7 functions in class NgramModel
5x4 points for correctly implementing 4 functions in class NgramModelWithInterpolation
Report (35 points):

20 points for your program description and critical analysis
15 points for presenting the requested supporting data
Grading (Masters/2071)
Parts 0-2 Code + Report (60 Points):

Code (40 Points)
4x7 points for correctly implementing 7 functions in class NgramModel
3x4 points for correctly implementing 4 functions in class NgramModelWithInterpolation
Report (20 Points)
15 points for your program description and critical analysis
5 points for presenting the requested supporting data
Part 3 Code + Report (40 Points):

Code (25 Points)
Implemented code to run your experiment as described and we don't have any major flaws in the code that undermine the experiment
Report (15 Points)
Report describing the experiment and results as outlined in Part 3
Part 0: Generating N-Grams
Write a function ngrams(c, text) that produces a list of all n-grams of that use c elements of context from the input text. Each n-gram should consist of a 2-element tuple (context, char), where the context is itself a c-length string comprised of the c characters preceding the current character. If c=1, then produce bigrams, if c=2, trigrams. The sentence should be padded with c ~ characters at the beginning (we’ve provided you with start_pad(c) for this purpose). If c=0, all contexts should be empty strings. You may assume that c≥0.

>>> ngrams(1, 'abc')
[('~', 'a'), ('a', 'b'), ('b', 'c')]

>>> ngrams(2, 'abc')
[('~~', 'a'), ('~a', 'b'), ('ab', 'c')]
We’ve also given you the function create_ngram_model(model_class, path, c, k) that will create and return an n-gram model trained on the entire file path provided.

Part 1: Creating an N-Gram Model
1.1 Build a N-gram Language Model
In this section, you will build a simple n-gram language model that can be used to generate random text resembling a source document. Your use of external code should be limited to built-in Python modules, which excludes, for example, NumPy and NLTK.

In the NgramModel class, write an initialization method __init__(self, c, k) which stores the context length c of the model and initializes any necessary internal variables. Then write a method get_vocab(self) that returns the vocab (this is the set of all characters used by this model).

Write a method update(self, text) which computes the n-grams for the input sentence and updates the internal counts. Also, write a method prob(self, context, char) which accepts an c-length string representing a context and a character, and returns the probability of that character occurring, given the preceding context. If you encounter a novel context, the probability of any given char should be 
 where 
 is the size of the vocab.

 >>> m = NgramModel(1, 0)
 >>> m.update('abab')
 >>> m.get_vocab()
 {'b', 'a'}
 >>> m.update('abcd')
 >>> m.get_vocab()
 {'b', 'a', 'c', 'd'}
 >>> m.prob('a', 'b')
 1.0
 >>> m.prob('~', 'c')
 0.0
 >>> m.prob('b', 'c')
 0.5
Write a method random_char(self, context) which returns a random character according to the probability distribution determined by the given context. Specifically, let 
 be the vocab, sorted according to Python’s natural lexicographic ordering, and let  be a random number between 0 and 1. Your method should return the character vi such that

.
You should use a single call to the random.random() function to generate 
.

 >>> m = NgramModel(0, 0)
 >>> m.update('abab')
 >>> m.update('abcd')
 >>> random.seed(1)
 >>> [m.random_char('') for i in range(25)]
 ['a', 'c', 'c', 'a', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'c', 'a', 'b', 'b', 'a', 'd', 'd', 'a', 'a', 'b', 'd', 'b', 'a']
In the NgramModel class, write a method random_text(self, length) which returns a string of characters chosen at random using the random_char(self, context) method. Your starting context should always be c ~ characters, and the context should be updated as characters are generated. If c=0, your context should always be the empty string. You should continue generating characters until you’ve produced the specified number of random characters, then return the full string.

 >>> m = NgramModel(1, 0)
 >>> m.update('abab')
 >>> m.update('abcd')
 >>> random.seed(1)
 >>> m.random_text(25)
 abcdbabcdabababcdddabcdba
1.2 Writing Shakespeare
Now you can train a language model using the training corpus of Shakespeare. Afterward, try generating some Shakespeare with different order n-gram models. For example, you can try using different n by running the following commands:

>>> m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
>>> m.random_text(250)

>>> m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
>>> m.random_text(250)

>>> m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
>>> m.random_text(250)

>>> m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
>>> m.random_text(250)
After generating a bunch of short passages, do you notice anything? What do you think of them? Are they as good as 1000 monkeys working at 1000 typewritersLinks to an external site.?  They all start with F! In fact, after we hit a certain order, the first word is always First? Why is that? Report your generated text and explain what is going on in your writeup. (Note regarding this output. If you do not see this pattern appearing, and you are sure you implemented the previous parts correctly, it likely has to do with the random numbers being produced. Setting your rng seed to be 1 should produce the results we are intending to highlight, but you shouldn't need to do this.)

Part 2: Perplexity, Smoothing, and Interpolation
In this part of the assignment, you’ll adapt your code in order to implement several of the techniques described in the Jurafsky and Martin textbookLinks to an external site..

2.1 Perplexity
How do we know whether a language model is good? There are two basic approaches:

Task-based evaluation (also known as extrinsic evaluation), where we use the language model as part of some other task, like automatic speech recognition, or spelling correction, or an OCR system that tries to covert a professor’s messy handwriting into text.
Intrinsic evaluation. Intrinsic evaluation tries to directly evaluate the goodness of the language model by seeing how well the probability distributions that it estimates are able to explain some previously unseen test set.
Here’s what the textbook says:

For an intrinsic evaluation of a language model we need a test set. As with many of the statistical models in our field, the probabilities of an N-gram model come from the corpus it is trained on, the training set or training corpus. We can then measure the quality of an N-gram model by its performance on some unseen data called the test set or test corpus. We will also sometimes call test sets and other datasets that are not in our training sets held out corpora because we hold them out from the training data.

So if we are given a corpus of text and want to compare two different N-gram models, we divide the data into training and test sets, train the parameters of both models on the training set, and then compare how well the two trained models fit the test set.

But what does it mean to “fit the test set”? The answer is simple: whichever model assigns a higher probability to the test set is a better model.

You’ll implement the most common method for the intrinsic metric of language models: perplexity. The perplexity of a language model on a test set is the inverse probability of the test set, normalized by the number of characters. For a test set 
:

Now implement the perplexity(self, text) function in NgramModel. A couple of things to keep in mind:

Numeric underflow is going to be a problem, so consider using logs.
Perplexity is undefined if the language model assigns any zero probabilities to the test set. In that case your code should return positive infinity - float('inf').
On your unsmoothed models, you’ll definitely get some zero probabilities for the test set. To test your code, you should try computing perplexity on the training set, and you should compute perplexity for your language models that use smoothing and interpolation.
# Examples to run your code
>>> m = NgramModel(1, 0)
>>> m.update('abab')
>>> m.update('abcd')
>>> m.perplexity('abcd')
1.189207115002721
>>> m.perplexity('abca')
inf
>>> m.perplexity('abcda')
1.515716566510398
We provide you two test files, a New York Times article, and several of Shakespeare’s sonnets. You need to report perplexity scores for each sentence (i.e., line) in the test document, as well as the document average. In addition, please critically analyze your results: e.g., what does your perplexity indicate in different test data? what does a comparison of different n (grams) in terms of perplexity tell you about? which performs best? why do you think your models performed the way they did? etc.

2.2 Smoothing
Laplace smoothing adds one to each count (hence its alternate name add-one smoothing). Since there are 
 characters in the vocabulary and each one was incremented, we also need to adjust the denominator to take into account the extra V observations.

A variant of Laplace smoothing is called Add-k smoothing or Add-epsilon smoothing. This is described in section Add-k 3.5.2 in the Jurafsky and Martin textbookLinks to an external site.. Update your add-k smoothing in the NgramModel from Part 1. Please try to test different k values and n-grams models on two of the test data provided. You can do it by fixing n and varying k or fixing k and vary n. Note that, you need to report perplexity scores for each sentence (i.e., line) in the test document, as well as the document average. After running your model, what did you observe? Is smoothed model better than unsmoothed model? Please report your findings in your writeup.

# Examples to run your code
>>> m = NgramModel(1, 1)
>>> m.update('abab')
>>> m.update('abcd')
>>> m.prob('a', 'a')
0.14285714285714285
>>> m.prob('a', 'b')
0.5714285714285714
>>> m.prob('c', 'd')
0.4
>>> m.prob('d', 'a')
0.25
2.3 Interpolation
The idea of interpolation is to calculate the higher-order n-gram probabilities also combining the probabilities for lower-order n-gram models. Like smoothing, this helps us avoid the problem of zeros if we haven’t observed the longer sequence in our training data. Here’s the math: , where .

We’ve provided you with another class definition NgramModelWithInterpolation that extends NgramModel for you to implement interpolation. If you’ve written your code robustly, you should only need to override the get_vocab(self), update(self, text), and prob(self, context, char) methods, along with the initializer.

The value of c passed into __init__(self, c, k) specifies the context length of the highest order n-gram model to be considered by the model (e.g. c=2, indicating trigrams, will consider 3 different length n-grams). Add-k smoothing should take place only when calculating the individual order n-gram probabilities, not when calculating the overall interpolation probability.

By default set the lambdas to be equal weights, but you should also write a helper function that can be called to overwrite this default. Setting the lambdas in the helper function can either be done heuristically or by using a development set, but in the example code below, we’ve used the default.

>>> m = NgramModelWithInterpolation(1, 0)
>>> m.update('abab')
>>> m.prob('a', 'a')
0.25
>>> m.prob('a', 'b')
0.75

>>> m = NgramModelWithInterpolation(2, 1)
>>> m.update('abab')
>>> m.update('abcd')
>>> m.prob('~a', 'b')
0.4682539682539682
>>> m.prob('ba', 'b')
0.4349206349206349
>>> m.prob('~c', 'd')
0.27222222222222225
>>> m.prob('bc', 'd')
0.3222222222222222
Please experiment with a few combinations of lambdas, values of k, and n (for n-grams) on two of the test data, e.g., fix n and k, vary lambda, or fix k and lambda, vary n, etc. Please report perplexity scores for each sentence (i.e., line) in the test document, as well as the document average on two of the test data. Discuss their effects in your writeup.

Part 3 Masters Component: Breaking Caesar Shift Encrypted Messages
I am trial running this part of the assignment with Masters Students because I am not sure how well it will go.

Caesar Shift is a widely known substitution cipher. The method for encrypting takes each letter in the plaintext and replaces it with a letter some fixed number of positions down the alphabet. One of the problems with single-alphabet substitution methods are that they are vulnerable to frequency attacks. What we mean is that even though e has shifted to b in a left shift of 3, it won't change how often e occurs in the English language, thus would be reflected in b. N-Gram Language models are completely built off of frequency information and so it may be possible to break an Caeser Shift encrypted message using one.

The goal of this portion of the assignment is explore how effectively we can break a Caeser Shifted message using N-Gram Models.

Provided to you are 3 things for this part of the assignment. A caesarShifter.py script that you may use to encrypt and decrypt messages as you need, a wikicorpus.tar file which contains over 18 thousand wikipedia articles with a lot of the extra formatting stripped from it, and a encryptedMessage.txt file you are tasked to try and decrypt.

I am asking you to do the following:

Design an experiment that could be used to decrypt Caesar Shifted messages:
Using the things we have learned and made regarding Ngrams, Perplexity, and Smoothing
This involves producing a hypothesis, testing it, and then discussing the results you get from your approach
Convince yourself using the provided caesarShifter.py script that your approach should work
Write up a report discussing your approach and results and where your approach might work well or work poorly
In your report tell me what you think the original message was and the key/shift value on the encryptedMessage.txt
I am more curious to see how close we can get to finding the original message so try your best to to retrieve it but I likely won't dock points if you don't find it.
It may be a good idea to update the Ngram to save the the model to disk to save on computation time.

On Writeups
Here is an announcement I posted last year on general tips on writing reports for this course.

Here is some general advice on how you should do pretty much all of the write ups for this course.
Something to keep in mind is the write ups are scientific experiments with an analysis of the results. So generally doing the following structure for each of the "things to report on" should be helpful, with examples from Assignment 2 part 1.2:

State the hypothesis: What are we exploring; what are we changing to handle this exploration; and what do we hope to see?

EX. 1.2 -> You are exploring the output of the Shakespeare N-gram model by varying the context size and trying to explain why the word "First" appears consistently.

Explain any assumptions that are being made as you explore the hypothesis

EX. 1.2 do you get the results by setting the random seed to 1 before printing. Do you make any considerations on how you treat your data. What data are you using.

Most decisions you make that aren't explicitly handled in the assignment is an assumption being made with respect to experiments.

Show and describe your results: At this point we have a result we want to show, show it and describe what we should be considering.

EX. 1.2 show the text you get and describe the desired trends you happen to see. Tell us that we should be looking for the word 'First' or any other trend you want us to look at.

When considering results, you might need to do several experiments to see the result, but only show enough results to be convincing, and this pertains more to the perplexity questions but still applies for all assignments.

Provide interpretations of the results. Describe the results in more detail. This might be exploring why or how the results came to be. What are the other implications of this result? Who might they affect?

Ex. 1.2 .... {I would give an example but that is the assignment so tell me, why do you think 'First' comes up in the output}

And that is pretty much it. So long as you use evidence-based reasoning, then you are not likely to be wrong when answering the questions or providing interpretations. The assignments are slow practice on writing a research paper covering the parts around making a hypothesis, exploring it, and then reporting what you found. You will also be doing this for the project, but there will be more you have to describe, but more on that when it is relevant.

 

Recommended Readings
Language Modeling with N-grams.Links to an external site. Dan Jurafsky and James H. Martin. Speech and Language Processing (3rd edition draft).
A Bit of Progress in Language ModelingLinks to an external site.. Joshua Goodman. Computer Speech and Language.
The Unreasonable Effectiveness of Character-level Language Models.Links to an external site. Yoav Goldberg. Response to Andrej Karpathy's blog post. 2015.
Language Independent Authorship Attribution using Character Level Language Models.Links to an external site. Fuchun Pen, Dale Schuurmans, Vlado Keselj, Shaojun Wan. EACL 2003.
Acknowledgment
This assignment is adapted from Prof. Mark Yatskar.

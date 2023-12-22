import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    result = []
    # text = text.lower()
    for i in range(len(text)):
        if i == 0:
            result.append((start_pad(c), text[i]))
            continue
        elif i <= c:
            result.append(((start_pad(c-i)+text[:i]), text[i]))
        else:
            result.append((text[i-c:i], text[i]))
    return result



def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

################################################################################
# Data collection methods for the writeup, feel free to ignore:
################################################################################

def do_writeup_add_to_file(model_class, path, c=2, k=0):
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        file_content = f.read() 
        output_file = open("results_writeup.txt", "a")
        output_file.write("Model Class: " + str(model_class) + "\n")
        output_file.write("File Name: " + str(path) + "\n")
        output_file.write("C: " + str(c) + "\n")
        output_file.write("K: " + str(k) + "\n")
        lines = file_content.split('\n')
        # print(str(lines))
        temp = 0.0
        count = 0
        for line in lines:
            model.update(line)
            if line == "":
                continue
            output_file.write("Perplexity of '" + str(line) + "' is " + str(model.perplexity(line)) + "\n")
            temp += model.perplexity(line)
            count += 1
        output_file.write("Total lines: " + str(count) + "\n")
        output_file.write("Average Perplexity: " + str(temp/count) + "\n")
        output_file.close()

def do_2dot3_writeup(model_class, path, c, k, l):
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        file_content = f.read() 
        title = ''
        file_name =''
        if path == 'test_data/shakespeare_sonnets.txt':
            title = '2.3 Shakespeare\'s Sonnets Perplexity with Interpolation'
            file_name = 'shakespeare_interpolated.txt'
        else:
            title = '2.3 New York Times Article Perplexity with Interpolation'
            file_name = 'nytimes_interpolated.txt'
        output_file = open(file_name, "a")
        output_file.write(title + "\n")
        output_file.write("\n")
        output_file.write("Model Class: " + str(model_class) + "\n")
        output_file.write("File Name: " + str(path) + "\n")
        output_file.write("C: " + str(c) + "\n")
        output_file.write("K: " + str(k) + "\n")
        output_file.write("L: " + str(l) + "\n")
        # model.update(file_content) 
        lines = file_content.split('\n')
        # print(str(lines))
        temp = 0.0
        count = 0
        for line in lines:
            model.set_lambdas(l)
            model.update(line)
            if line == "":
                continue
            output_file.write("Perplexity of '" + str(line) + "' is " + str(model.perplexity(line)) + "\n")
            temp += model.perplexity(line)
            count += 1
        output_file.write("Total lines: " + str(count) + "\n")
        output_file.write("Average Perplexity: " + str(temp/count) + "\n")
        output_file.write("\n")
        output_file.close()

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''
    c = ''
    k = ''
    text = ''
    ngrams = []
    characterFreq = {}
    context_counts = {} # in the case of bigrams, how many times is a character a context
    counts = {} # 
    def __init__(self, c, k):
        self.c = c
        self.k = k
    
    def set_k(self, k):
        self.k = k

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        # Update vocab
        for c in self.text:
            if self.characterFreq.__contains__(c):
                self.characterFreq[c] += 1
            else:
                self.characterFreq[c] = 1
        return set(self.characterFreq.keys())

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        # self.context_counts = {} #empty
        # self.counts = {} #empty
        self.text = text
        self.ngrams = ngrams(self.c, text)
        for ngram in self.ngrams:
            if ngram[0] not in self.context_counts:
                self.context_counts[ngram[0]] = 0
            self.context_counts[ngram[0]] += 1
        for ngram in self.ngrams:
            if ngram[0] not in self.counts:
                self.counts[ngram[0]] = {}
            if ngram[1] not in self.counts[ngram[0]]:
                self.counts[ngram[0]][ngram[1]] = 0
            self.counts[ngram[0]][ngram[1]] += 1
        

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        # context = context.lower()
        if context not in self.context_counts:
            return 1.0 / len(self.get_vocab())
        context_count = self.context_counts[context]
        # Smoothing, so not needed
        # if char not in self.counts.get(context, {}):
        #     return 0.0
        char_count = self.counts[context].get(char, 0)
        # Pre smoothing
        # return char_count / context_count
        # Post smoothing
        return (char_count + self.k) / (context_count + (self.k * len(self.get_vocab())))
        
    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        # context = context.lower()
        rand = random.random()
        if context not in self.counts:
            return random.choice(sorted(self.get_vocab())) #fallback if novel context, per class discussion
        for i in self.counts[context]:
            rand -= self.prob(context, i)
            if rand <= 0:
                return i
        return None

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = ''
        for i in range(length):
            if i == 0:
                # print(str(start_pad(self.c)))
                text += self.random_char(start_pad(self.c))
            elif i <= self.c:
                # print(start_pad((self.c-i))+text[:i])
                text += self.random_char(str(start_pad((self.c-i))+text[:i]))
            else:
                # print("here " + text[-self.c:])
                text += self.random_char(text[-self.c:])
        return text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        list_ngrams = ngrams(self.c, text)
        # print(str(list_ngrams))
        log_prob_sum = 0
        for (context, char) in list_ngrams:
            prob = self.prob(context, char)
            if prob == 0:
                return float('inf')
            else:
                log_prob_sum += math.log(prob)
        # print("log_prob_sum" + str(log_prob_sum))
        # print("text = " + text)
        perplexity = math.exp(-1/(len(text)) * log_prob_sum)
        return perplexity


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        self.c = c 
        self.models = {} 
        self.lambdas = [1.0 / (c+1)] * (c+1) #equal lambdas at start
        for order in range(c + 1):
            # unigram to c-gram
            self.models[order] = NgramModel(order, k)
            self.lambdas[order] = 1/(c + 1)
        self.k = k
    
    def set_lambdas(self, lambdas):
        # sum of all lambdas must equal 1 - error checking was a bit of trouble because of python's floating point precision problems
        # So I'll trust the user to give good lambda values - a user would never intentionally give bad values, right? 
        # Who needs input sanitzation anyway?
        self.lambdas = lambdas
    

    def get_vocab(self):
        vocab = set()
        for order in range(self.c + 1):
            model = self.models[order]
            vocab = vocab.union(model.get_vocab()) 
        return vocab

    def update(self, text):
        for order in range(self.c + 1):
            model = self.models[order]
            model.update(text)

    def prob(self, context, char):
        prob = 0
        for order in range(self.c + 1):
            model = self.models[order]
            my_lambda = self.lambdas[order]
            if model.c == 0:
                temp = ''
            else:
                temp = context[-model.c:]
            prob += my_lambda * model.prob(temp, char)
        return prob








# Only execute these when running this file, not from an autograder.


# if __name__ == '__main__':
#     m = create_ngram_model(NgramModel, 'train/shakespeare_input.txt', 3)
#     # print(m.ngrams)
#     random.seed(1)
#     print(m.random_text(250))
#     print("\n")


# if __name__ == '__main__':
#     m = NgramModel(1, 0)
#     m.update('abab')
#     m.update('abcd')
#     print(str(m.get_vocab()))
#     print(str(m.perplexity('abcd')))
#     print(str(m.perplexity('abca')))
#     print(str(m.perplexity('abcda')))

# if __name__ == '__main__':
#     m = NgramModel(1, 1)
#     m.update('abab')
#     m.update('abcd')
#     print(str(m.get_vocab()))
#     print(str(m.perplexity('abcd')))
#     print(str(m.perplexity('abca')))
#     print(str(m.perplexity('abcda')))

# 2.2 Writeup Section to automate reports for each line of perplexity, smoothing and interpolation
# if __name__ == '__main__':
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 2)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 2, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 2, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 3)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 3, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 3, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 4)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 4, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 4, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 7)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 7, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/nytimes_article.txt', 7, 1)

#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 2)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 2, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 2, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 3)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 3, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 3, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 4)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 4, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 4, 1)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 7)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 7, 0.5)
#     do_writeup_add_to_file(NgramModel, 'test_data/shakespeare_sonnets.txt', 7, 1)

# if __name__ == '__main__':
#     m = NgramModelWithInterpolation(1, 0)
#     m.update('abab')
#     print(str(m.prob('a', 'a')))
#     print(str(m.prob('a', 'b')))
#     m1 = NgramModelWithInterpolation(2, 1)
#     m1.update('abab')
#     m1.update('abcd')
#     print(str(m1.prob('~a', 'b')))
#     print(str(m1.prob('ba', 'b')))
#     print(str(m1.prob('~c', 'd')))
#     print(str(m1.prob('bc', 'd')))


# 2.3 Writeup Section to automate reports for each line of perplexity, smoothing and interpolation
# if __name__ == '__main__':
#     # Shakespeare
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 0, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 0, [0.5, 0.33, 0.17])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 0.5, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 0.5, [0.5, 0.33, 0.17])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 1, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 2, 1, [0.5, 0.33, 0.17])


#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 0, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 0, [0.3, 0.3, 0.2, 0.2])
    
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 0.5, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 0.5, [0.3, 0.3, 0.2, 0.2])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 1, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 3, 1, [0.3, 0.3, 0.2, 0.2])


#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 0, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 0, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 0.5, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 0.5, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 1, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 4, 1, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 0, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 0, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 0.5, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 0.5, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 1, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/shakespeare_sonnets.txt', 7, 1, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

#     # New York Times
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 0, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 0, [0.5, 0.33, 0.17])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 0.5, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 0.5, [0.5, 0.33, 0.17])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 1, [0.33, 0.33, 0.33])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 2, 1, [0.5, 0.33, 0.17])


#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 0, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 0, [0.3, 0.3, 0.2, 0.2])
    
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 0.5, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 0.5, [0.3, 0.3, 0.2, 0.2])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 1, [0.25, 0.25, 0.25, 0.25])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 3, 1, [0.3, 0.3, 0.2, 0.2])


#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 0, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 0, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 0.5, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 0.5, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 1, [0.2, 0.2, 0.2, 0.2, 0.2])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 4, 1, [0.5, 0.2, 0.1, 0.1, 0.1])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 0, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 0, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 0.5, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 0.5, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 1, [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     do_2dot3_writeup(NgramModelWithInterpolation, 'test_data/nytimes_article.txt', 7, 1, [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05])

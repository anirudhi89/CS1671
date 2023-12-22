# Name: Anirudh Iyer

'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Canvas.
'''

import re

def check_for_foo_or_bar(text):

   '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
   '''

   text = text.lower()
   text = text.strip()
   return re.search(r'\bfoo\b', text) and re.search(r'\bbar\b', text)


def replace_rgb(text):

   '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''
   pattern = r'(?:(?<=\s)|(?<=^))(#(?:[0-9A-Fa-f]{3}){1,2}|rgb\(\s*(?:(?:\d+\.\d+|\d+)\s*,\s*){2}(?:\d+\.\d+|\d+)\s*\))(?=\s|$)'
   result = re.sub(pattern, 'COLOR', text)

   return result

   


def wine_text_processing(wine_file_path, stopwords_file_path):
   '''Process the two files to answer the following questions and output results to stdout.

   1. What is the distribution over star ratings?
   2. What are the 10 most common words used across all of the reviews, and how many times
      is each used?
   3. How many times does the word 'a' appear?
   4. How many times does the word 'fruit' appear?
   5. How many times does the word 'mineral' appear?
   6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
      In natural language processing, we call these common words "stop words" and often
      remove them before we process text. stopwords.txt gives you a list of some very
      common words. Remove these stopwords from your reviews. Also, try converting all the
      words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
      different words). Now what are the 10 most common words across all of the reviews,
      and how many times is each used?
   7. You should continue to use the preprocessed reviews for the following questions
      (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
      reviews, and how many times is each used? 
   8. What are the 10 most used words among the 1 star reviews, and how many times is
      each used? 
   9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
      "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
      "white" reviews?
   10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
         reviews?

   No return value.
   '''
   star_rating = []
   formattedOutputQ1 = ""
   formattedOutputQ2 = ""
   formattedOutputQ3 = ""
   formattedOutputQ4 = ""
   formattedOutputQ5 = ""
   formattedOutputQ6 = ""
   formattedOutputQ7 = ""
   formattedOutputQ8 = ""
   formattedOutputQ9 = ""
   formattedOutputQ10 = ""


   # Question 1
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      
      for line in wine_file:
         line = line.lower()
         #search for text after the tab
         star_rating.append(re.search(r'(?<=\t).+', line).group(0))
         star_rating.sort()
         formattedOutputQ1 = "******\t" + str(star_rating.count('******')) + '\n' "*****\t" + str(star_rating.count('*****')) + '\n' + "****\t" + str(star_rating.count('****')) + '\n' + "***\t" + str(star_rating.count('***')) + '\n' + "**\t" + str(star_rating.count('**')) + '\n' + "*\t" + str(star_rating.count('*')) + '\n'

   # Question 2
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      word_counts = {}
      for line in wine_file:
         beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
         #word = seperated by space or hyphen
         sentence = re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)

         for word in sentence:
            if word in word_counts:
                  word_counts[word] += 1
            else:
                  word_counts[word] = 1
      sorted_word_counts = sorted(word_counts.items(), key=lambda x:x[1], reverse=True)
      dict_sorted_word_counts = dict(sorted_word_counts)
      
      i = 0
      for k, v in dict_sorted_word_counts.items():
         formattedOutputQ2 += k + "\t" + str(v) + '\n'
         i += 1
         if i == 10:
            break
   
   # Question 3
   q3count=0
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:  
      for line in wine_file:
         line=line.lower()
         for m in re.findall(r'\ba\b', line):
            q3count=q3count+1
   formattedOutputQ3 = str(q3count)

   # Question 4
   q4count=0
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:  
      for line in wine_file:
         line=line.lower()
         for m in re.findall(r'\bfruit\b', line):
            q4count=q4count+1
   formattedOutputQ4 = str(q4count)

   # Question 5
   q4count=0
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:  
      for line in wine_file:
         line=line.lower()
         for m in re.findall(r'\bmineral\b', line):
            q4count=q4count+1
   formattedOutputQ5 = str(q4count)

   # Question 6
   q6sentence = []
   q6word_counts = {}
   stopwords = {}
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      for line in wine_file:
         line = line.lower()
         beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
         #word = seperated by space or hyphen
         q6sentence += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
         # print(q6sentence)

   with open(stopwords_file_path, encoding='ISO-8859-1') as stopwords_file:
      stopwords = set(stopwords_file.read().split())
      stopwords = [word.lower() for word in stopwords]
      # print(stopwords)
      # print(q6sentence)
      for word in q6sentence:
         if word in stopwords:
            continue
         elif word in q6word_counts:
               q6word_counts[word] += 1
         else:
               q6word_counts[word] = 1
   sorted_q6word_counts = sorted(q6word_counts.items(), key=lambda x:x[1], reverse=True)
   dict_sorted_q6word_counts = dict(sorted_q6word_counts)  
   i = 0
   for k, v in dict_sorted_q6word_counts.items():
      formattedOutputQ6 += k + "\t" + str(v) + '\n'
      i += 1
      if i == 10:
         break


   # Question 7
   stopwords = {}
   five_star_reviews = []
   with open(stopwords_file_path, encoding='ISO-8859-1') as stopwords_file:
      stopwords = set(stopwords_file.read().split())
      stopwords = [word.lower() for word in stopwords]
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      for line in wine_file:
            line = line.lower()
            # if ating is *****, add review to list
            if re.search(r'(?<=\t).+', line).group(0) == '*****':
               beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
               #word = seperated by space or hyphen
               five_star_reviews += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
   # print(five_star_reviews)
   q7_word_counts = {}
   for word in five_star_reviews:
      if word not in stopwords:
         if word in q7_word_counts:
               q7_word_counts[word] += 1
         else:
               q7_word_counts[word] = 1
   # print(q7_word_counts)
   sorted_q7_word_counts = sorted(q7_word_counts.items(), key=lambda x:x[1], reverse=True)
   dict_sorted_q7_word_counts = dict(sorted_q7_word_counts)
   i = 0
   for k, v in dict_sorted_q7_word_counts.items():
      formattedOutputQ7 += k + "\t" + str(v) + '\n'
      i += 1
      if i == 10:
         break
   
   # Question 8
   stopwords = {}
   one_star_reviews = []
   with open(stopwords_file_path, encoding='ISO-8859-1') as stopwords_file:
      stopwords = set(stopwords_file.read().split())
      stopwords = [word.lower() for word in stopwords]
   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      for line in wine_file:
            line = line.lower()
            # if ating is *****, add review to list
            if re.search(r'(?<=\t).+', line).group(0) == '*':
               beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
               #word = seperated by space or hyphen
               one_star_reviews += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
   # print(one_star_reviews)
   q8_word_counts = {}
   for word in one_star_reviews:
      if word not in stopwords:
         if word in q8_word_counts:
               q8_word_counts[word] += 1
         else:
               q8_word_counts[word] = 1
   # print(q8_word_counts)
   sorted_q8_word_counts = sorted(q8_word_counts.items(), key=lambda x:x[1], reverse=True)
   dict_sorted_q8_word_counts = dict(sorted_q8_word_counts)
   i = 0
   for k, v in dict_sorted_q8_word_counts.items():
      formattedOutputQ8 += k + "\t" + str(v) + '\n'
      i += 1
      if i == 10:
         break
   # print(formattedOutputQ8)


   # Question 9
   stopwords = {}
   red_q9sentence = []
   white_q9sentence = []
   with open(stopwords_file_path, 'r') as stopwords_file:
      stopwords = set(stopwords_file.read().split())
      stopwords = [word.lower() for word in stopwords]

   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      for line in wine_file:
            line = line.lower()
            beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
            # if review has the word red, add to red list
            if re.search(r'\bred\b', beforetab):
               red_q9sentence += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
            # if reivew has the word white, add to white list.
            if re.search(r'\bwhite\b', beforetab):
               white_q9sentence += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
            
   # find top 10 words in red list that are not in white list
   q9_word_counts = {}
   for word in red_q9sentence:
      if word not in stopwords:
         if word not in white_q9sentence:
            if word in q9_word_counts:
                  q9_word_counts[word] += 1
            else:
                  q9_word_counts[word] = 1
   sorted_q9_word_counts = sorted(q9_word_counts.items(), key=lambda x:x[1], reverse=True)
   dict_sorted_q9_word_counts = dict(sorted_q9_word_counts)
   i = 0
   for k, v in dict_sorted_q9_word_counts.items():
      formattedOutputQ9 += k + "\t" + str(v) + '\n'
      i += 1
      if i == 10:
         break
   # print(formattedOutputQ9)

   # Question 10
   stopwords = {}
   red_q10sentence = []
   white_q10sentence = []
   with open(stopwords_file_path, 'r') as stopwords_file:
      stopwords = set(stopwords_file.read().split())
      stopwords = [word.lower() for word in stopwords]

   with open(wine_file_path, encoding='ISO-8859-1') as wine_file:
      for line in wine_file:
            line = line.lower()
            beforetab = re.search(r'^(.*?)(?=\t|$)', line).group(0)
            # if review has the word red, add to red list
            if re.search(r'\bred\b', beforetab):
               red_q10sentence += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
            # if reivew has the word white, add to white list.
            if re.search(r'\bwhite\b', beforetab):
               white_q10sentence += re.findall(r'\w+(?:-\w+|-)*|\$(?:\d+\.\d+|\d+)|[^.,\s()+]+', beforetab)
            
   # find top 10 words in red list that are not in white list
   q10_word_counts = {}
   for word in white_q10sentence:
      if word not in stopwords:
         if word not in red_q10sentence:
            if word in q10_word_counts:
                  q10_word_counts[word] += 1
            else:
                  q10_word_counts[word] = 1
   sorted_q10_word_counts = sorted(q10_word_counts.items(), key=lambda x:x[1], reverse=True)
   dict_sorted_q10_word_counts = dict(sorted_q10_word_counts)
   i = 0
   for k, v in dict_sorted_q10_word_counts.items():
      formattedOutputQ10 += k + "\t" + str(v) + '\n'
      i += 1
      if i == 10:
         break
   # print(formattedOutputQ10)


   print("Question 1 outputs: \n" + formattedOutputQ1)
   print("Question 2 outputs: " + "\n" + formattedOutputQ2)
   print("Question 3 outputs: " + "\n" + formattedOutputQ3)
   print("Question 4 outputs: " + "\n" + formattedOutputQ4)
   print("Question 5 outputs: " + "\n" + formattedOutputQ5)
   print("Question 6 outputs: " + "\n" + formattedOutputQ6)
   print("Question 7 outputs: " + "\n" + formattedOutputQ7)
   print("Question 8 outputs: " + "\n" + formattedOutputQ8)
   print("Question 9 outputs: " + "\n" + formattedOutputQ9)
   print("Question 10 outputs: " + "\n" + formattedOutputQ10)


'''
Testing
'''
def test_suite():
   test_check_for_foo_or_bar()

   test_replace_rgb()
   
   test_wine_text_processing()


# TESTS

def test_check_for_foo_or_bar():
      #Expected false
   if (not check_for_foo_or_bar('foobart')):
      print ("Passed")
   else:
      print ("Failed")
   
   #Expected true
   if (check_for_foo_or_bar('foo bar')):
      print("Passed") #true
   else:
      print("Failed")

def test_replace_rgb():
      # replace_rgb("I like rgb(1, 2, 3) and rgb(2, 3, 4)."):
      expected = "I like COLOR and rgb(2, 3, 4)."
      print("Passed" if (expected == replace_rgb(("I like rgb(1, 2, 3) and rgb(2, 3, 4)."))) else "Failed")

def test_wine_text_processing():
   wine_filepath = 'data/wine.txt'
   stopwords_filepath = 'data/stopwords.txt'
   wine_text_processing(wine_filepath, stopwords_filepath)

# Only run the following lines of code if this file is being run directly - ie not on autograder
if __name__ == '__main__':
   test_suite()
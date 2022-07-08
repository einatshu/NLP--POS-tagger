"""
intro2nlp, assignment 4, 2020

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""
from math import log, isfinite
from collections import Counter
import random
import pandas as pd
import re

tagged_word_index = 0
tag_index = 1

import sys, os, time, platform, nltk


def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

allTagCounts = Counter()
allWordsCount = Counter()

all_capital = Counter()
start_with_capital = Counter()
singelton_tt = {}
singelton_tw = {}
suffix_options_list = ['ation',
                        'ible', 'ious', 'ment', 'ness', 'sion', 'ship', 'able','less','ward', 'wise', 'eer','cian','able',
                       'ion','ies', 'ity','off', 'ous', 'ive', 'ant', 'ary', 'ful', 'ing', 'ize', 'ise','est','ess', 'ate'
                       ,'al', 'er','or', 'ic', 'ed','es','th', 'en', 'ly', 'y', 's']
suffixes = {}

# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities

smooth_alpa = 0.01


def get_tags_list(allTagCounts):
    return list(set(allTagCounts.keys()).difference(set([END, START])))


def update_dic_counter(dic, key, value):
    key = key[0]
    if key not in dic:
        dic[key] = Counter()
    dic[key].update(value)


def get_laplace_smooth(numerator_counter, denominator_counter, vocabulary_size):
    return (numerator_counter+smooth_alpa)/(denominator_counter+smooth_alpa*vocabulary_size)


def is_word_start_with_capital(word, prev_tag):
    if word[0].isupper() and prev_tag != START:
        return True
    return False


def is_all_capital_word(word):
    regex = r"\b[A-Z][A-Z]+\b"
    matches = [i for i in re.finditer(regex, word)]
    if len(matches) > 0:
        return True
    return False

def learn(tagged_sentences):
    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts

    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by read_tagged_corpus().

    Returns:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
    """

    for suffix in suffix_options_list:
        suffixes[suffix] = Counter()
    for tagged_sentence in tagged_sentences:
        previous_tag = [START]
        allTagCounts.update(previous_tag)
        for tagged_word in tagged_sentence:
            current_word = [tagged_word[tagged_word_index]]
            current_tag = [tagged_word[tag_index]]

            # count capital letters stats
            #  count tags distribution of words of capital letters:
            if is_all_capital_word(current_word[0]):
                all_capital.update(current_tag)
            #  count tags distribution of words start with capital letters:
            else:
                if is_word_start_with_capital(current_word[0], previous_tag[0]):
                    start_with_capital.update(current_tag)
            allTagCounts.update(current_tag)
            allWordsCount.update(current_word)
            update_dic_counter(perWordTagCounts, current_word, current_tag)
            update_dic_counter(transitionCounts, previous_tag, current_tag)
            update_dic_counter(emissionCounts, current_tag, current_word)
            #  count word suffix
            for suffix in suffix_options_list:
                if current_word[0].endswith(suffix):
                    suffixes[suffix].update(current_tag)
                    break
            previous_tag = current_tag
        current_tag = [END]
        allTagCounts.update(current_tag)
        update_dic_counter(transitionCounts, previous_tag, current_tag)

    # count singeltones
    for tag in allTagCounts.keys():
        if tag != END:
            singelton_tt[tag] = [i for i in transitionCounts[tag].values()].count(1)
        if tag != START and tag != END:
            singelton_tw[tag] = [i for i in emissionCounts[tag].values()].count(1)

    for previous_tag in allTagCounts.keys():
        if previous_tag != END:
            for current_tag in transitionCounts[previous_tag].keys():
                A[(previous_tag, current_tag)] = get_value_from_transition(transitionCounts[previous_tag][current_tag],
                                                                           previous_tag,
                                                                           current_tag)
            if previous_tag != START:
                tag = previous_tag
                for current_word in emissionCounts[tag].keys():
                    B[(tag, current_word)] = get_value_from_emmission(emissionCounts[tag][current_word], tag, current_word)

    return [allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B]


def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
      (same index) in the input sentence. Each word is tagged by the tag most
      frequently associated with it. OOV words are tagged by sampling from the
      distribution of all tags.

      Args:
          sentence (list): a list of tokens (the sentence to tag)
          perWordTagCounts (Counter): tags per word as specified in learn()
          allTagCounts (Counter): tag counts, as specified in learn()

      Return:
          list: list of pairs

      """
    sentence_tags = []
    for word in sentence:
        # best_tag = allTagCounts.most_common(1)[0][0]
        best_tag = random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1)[0]
        if word in perWordTagCounts:
            best_tag = perWordTagCounts[word].most_common(1)[0][0]
        # else:
        #     if word.lower() in perWordTagCounts:
        #         best_tag = perWordTagCounts[word.lower()].most_common(1)[0][0]
        sentence_tags.append((word, best_tag))
    return sentence_tags


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterby
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        list: list of pairs
    """
    tags = viterbi(sentence, A, B)
    tagged_sentence = []
    for index, word in enumerate(sentence):
        tagged_sentence.append([word, tags[index]])
    return tagged_sentence


def get_value_from_transition(transition_count, previous_tag, current_tag):
    """
        calc the transition log prob, use One-Count Smoothing to smooth to prob for open groups of
        tags and closed groups of tags.
    :param transition_count: as calc on training
    :param previous_tag:
    :param current_tag:
    :return: transition log prob
    """
    ptt_backof = (allTagCounts[current_tag] / sum(allWordsCount.values()))
    singelton_count = 1 + singelton_tt[previous_tag]
    value= log(((transition_count + ptt_backof*singelton_count) / (allTagCounts[previous_tag] +singelton_count)))
    return value


def get_word_suffix(word):
    """
    :param word:
    :return:  return the suffix for the given word
    """
    for suffix in suffix_options_list:
        if word.endswith(suffix) and word != suffix:
            return suffix
    return ''


def get_word_features_weight(word, current_tag, prev_tag):
    """
        for oov words- calc word features for suffix and capital letters
    :param word:
    :param current_tag:
    :param prev_tag: used for calc weight for first letter as capital letter, don't add weight if prev_tag is START
    :return: word features sum
    """
    word_suffix = get_word_suffix(word.lower())
    word_suffix_weight = 0
    word_all_capital_weight = 0
    word_start_with_capital_weight = 0
    if word_suffix != '':
        word_suffix_weight = max(suffixes[word_suffix][current_tag], word_suffix_weight)
    if is_all_capital_word(word):
        word_all_capital_weight = max(all_capital[current_tag], word_all_capital_weight)
    else:
        if is_word_start_with_capital(word, prev_tag):
            word_start_with_capital_weight = max(start_with_capital[current_tag], word_start_with_capital_weight)
    return word_suffix_weight+word_start_with_capital_weight+word_all_capital_weight


def get_value_from_emmission(emmission_count, current_tag, word, prev_tag=None):
    """
        return the emission logged prob for the given tag and word, if the word are oov then add to consideration
        word features as suffix and capital letters, I used One-Count Smoothing to smooth to prob for open groups of
        tags and closed groups of tags.

    :param emmission_count: number of appearances of word with tag in training set
    :param current_tag: tag to evaluate
    :param word: word to evaluate
    :param prev_tag: used only for oov words, for word feature weight
    :return: the emission logged prob
    """
    word_features_weight = 1
    if prev_tag is not None:
        word_features_weight = max(word_features_weight, get_word_features_weight(word, current_tag, prev_tag))
    ptw_backof = ((allWordsCount[word] + 1) / (sum(allWordsCount.values()) + len(perWordTagCounts)))
    singelton_count = 1 + singelton_tw[current_tag]
    value = log(((emmission_count + ptw_backof * singelton_count * word_features_weight) / (allTagCounts[current_tag] + singelton_count)))
    return value


def get_value_from_A(A, previous_tag, current_tag):
    """
        wrapper the access to A, if the transition is known from training then return value from A,
        else calc the transition online
    :param A: matrix A
    :param previous_tag:
    :param current_tag:
    :return: log transition prob
    """
    if (previous_tag, current_tag) in A:
        return A[(previous_tag, current_tag)]
    else:
        return get_value_from_transition(0, previous_tag, current_tag)


def get_value_from_B(B, current_tag, word, prev_tag):
    """
        wrapper the access to B, if the emission is known from training then return value from B,
        else calc the emission online
    :param B: matrix B
    :param current_tag:
    :param word:
    :param prev_tag:
    :return: log emission prob
    """
    if (current_tag, word) in B:
        return B[(current_tag, word)]
    return get_value_from_emmission(0, current_tag, word, prev_tag)


def get_most_probable_path(viterbi, A, B, prev_word, word, current_tag):
    """
        return the most probable path
    :param viterbi:
    :param A:
    :param B:
    :param prev_word:
    :param word:
    :param current_tag:
    :return:
    """
    possible_tags_prob = {}
    all_possible_previous_states = viterbi[viterbi[prev_word] != 0]
    for previous_tag, row in all_possible_previous_states.iterrows():
        possible_tags_prob[previous_tag] = row[prev_word][-1] + get_value_from_A(A, previous_tag, current_tag) +\
                                           get_value_from_B(B, current_tag, word, previous_tag)
    value = max(possible_tags_prob.items(), key=lambda k: k[1])
    return value


def viterbi(sentence, A, B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

    """
    # Hint 1: For efficiency reasons - for words seen in training there is no
    #      need to consider all tags in the tagset, but only tags seen with that
    #      word. For OOV you have to consider all tags.
    # Hint 2: start with a dummy item  with the START tag (what would it log-prob be?).
    #         current list = [ the dummy item ]
    # Hint 3: end the sequence with a dummy: the highest-scoring item with the tag END
    number_of_rows = len(get_tags_list(allTagCounts))
    number_of_columns = len(sentence)
    data = [[0 for i in range(number_of_columns)] for j in range(number_of_rows)]
    viterbi = pd.DataFrame(data, index=[k for k in get_tags_list(allTagCounts)], columns=[word+"_{}".format(index) for index, word in enumerate(sentence)], dtype=object)
    # initialization step
    intialize_states = get_tags_list(allTagCounts)
    first_word = sentence[0]
    if first_word in perWordTagCounts:
        intialize_states = perWordTagCounts[first_word].keys()

    for state in intialize_states:
        backpointer = START
        prob = get_value_from_B(B, state, first_word,START) +\
               get_value_from_A(A, START, state)
        viterbi.loc[state, sentence[0]+"_{}".format(0)] = (state, backpointer, prob)
    # recursion step
    for index, word in enumerate(sentence[1:]):
        possible_states = get_tags_list(allTagCounts)
        if word in perWordTagCounts:
            possible_states = perWordTagCounts[word].keys()

        for state in possible_states:
            backpointer, prob = get_most_probable_path(viterbi, A, B, sentence[index]+"_{}".format(index), word, state)
            viterbi.loc[state, word+"_{}".format(index+1)] = (state, backpointer, prob)
    word = sentence[-1]
    word_index = len(sentence)-1
    current_tag = END
    possible_tags_prob = {}
    all_possible_previous_states = viterbi[viterbi[word+"_{}".format(word_index)] != 0]
    for previous_tag, row in all_possible_previous_states.iterrows():
        possible_tags_prob[previous_tag] = row[word+"_{}".format(word_index)][-1] +\
                                           get_value_from_A(A, previous_tag, current_tag)
    backpointer, prob = max(possible_tags_prob.items(), key=lambda k: k[1])

    tags = []
    tags.append(END)
    # words_in_reversed_order = [re.sub('\_[0123456789]*$', '', column_name) for column_name in viterbi.columns.values]
    # words_in_reversed_order.reverse()
    for index, word in enumerate(reversed(viterbi.columns)):
        tags.append(backpointer)
        # tags.append((words_in_reversed_order[index], backpointer))
        word_possible_tags = viterbi[word]
        backpointer = word_possible_tags[backpointer][1]
    tags.reverse()
    return tags


def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): tthe HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags
    previous_tag = START
    for pair in sentence:
        word = pair[0]
        tag = pair[1]
        p += get_value_from_B(B, tag, word,previous_tag) + get_value_from_A(A, previous_tag, tag)
        previous_tag = tag
    tag = END
    p += get_value_from_A(A, previous_tag, tag)
    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


def get_number_of_tags_from_sentences(sentences):
    return sum([1 for sentence in sentences for pair in sentence])
# return sum([1 for pair in sentence for sentence in sentences])


def count_correct_for_test_sentences(gold_sentences, pred_sentences):
    assert len(gold_sentences) == len(pred_sentences)
    correct = 0
    correctOOV = 0
    OOV = 0

    number_of_tags_to_predict = get_number_of_tags_from_sentences(gold_sentences)
    for index, pred_sentence in enumerate(pred_sentences):
        correct_sentence, correctOOV_sentence, OOV_sentence =count_correct(gold_sentences[index], pred_sentence)
        correct += correct_sentence
        correctOOV += correctOOV_sentence
        OOV += OOV_sentence

    return correct, correctOOV, OOV, number_of_tags_to_predict


def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correcttly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """

    assert len(gold_sentence) == len(pred_sentence)
    correct = 0
    correctOOV = 0
    OOV = 0
    for index, pair in enumerate(pred_sentence):
        word = pair[0]
        predicted_tag = pair[1]
        label_tag = gold_sentence[index][1]
        # correct prediction
        if predicted_tag == label_tag:
            correct += 1
        #  oov count
        if word not in perWordTagCounts:
            OOV += 1
        if predicted_tag == label_tag and word not in perWordTagCounts:
            correctOOV += 1
    # if correct != len(pred_sentence):
    #     print("label:  {} \n predict: {}\n".format(gold_sentence, pred_sentence))
    return correct, correctOOV, OOV

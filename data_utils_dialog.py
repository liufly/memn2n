from __future__ import absolute_import

import os
import re
import numpy as np
from _collections import defaultdict

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    dev_file = [f for f in files if s in f and 'dev' in f][0]
    test_file = [f for f in files if s in f and 'tst' in f and 'OOV' not in f][0]
    train_data = get_stories(train_file, only_supporting)
    dev_data = get_stories(dev_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    if task_id != 6:
        testOOV_file = [f for f in files if s in f and 'tst' in f and 'OOV' in f][0]
        testOOV_data = get_stories(testOOV_file, only_supporting)
        return train_data, dev_data, test_data, testOOV_data
    else:
        return train_data, dev_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
#     return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]
    return sent.strip().split(' ')


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
#         line = str.lower(line)
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = a.split(' ')
            
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, sentence_size, memory_size, ans2idx):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

#         y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
#         for a in answer:
#             y[word_idx[a]] = 1
        a = ' '.join(answer)
        assert a in ans2idx
        y = np.zeros(len(ans2idx))
        y[ans2idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)

def vectorize_data_one_hot(data, word_idx, sentence_size, memory_size, ans2idx):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    vocab_size = len(word_idx) + 1 # 0 is reserved for nil word
    mem_vocab_size = memory_size + 1 # 0 is reserved for memory padding
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            sent = []
            for w in sentence:
                one_hot = [0] * vocab_size
                one_hot[word_idx[w]] = 1
                sent.append(one_hot)
            # ss.append([word_idx[w] for w in sentence] + [0] * ls)
            ss.append(sent + [[0] * vocab_size] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            # ss.append([0] * sentence_size)
            ss.append([[0] * vocab_size] * sentence_size)

        lq = max(0, sentence_size - len(query))
        # q = [word_idx[w] for w in query] + [0] * lq
        q = []
        for w in query:
            one_hot = [0] * vocab_size
            one_hot[word_idx[w]] = 1
            q.append(one_hot)
        q.extend([[0] * vocab_size] * lq)

#         y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
#         for a in answer:
#             y[word_idx[a]] = 1
        a = ' '.join(answer)
        assert a in ans2idx
        y = np.zeros(len(ans2idx))
        y[ans2idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)

def parse_kb(in_file):
    ret = defaultdict(set)
    with open(in_file) as f:
        for line in f:
            line = line.strip()
            attrs = line.split('\t')
            assert len(attrs) == 2
            kb_type = attrs[0].split(' ')[-1]
            assert kb_type[:2] == 'R_'
            value = attrs[1]
            ret[kb_type].add(value)
    return ret
            

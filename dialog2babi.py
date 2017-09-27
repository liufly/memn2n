'''
Created on 4Jan.,2017

@author: fei
'''

import os

def parse_dialog(in_file):
    stories = []
    story = []
    with open(in_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                stories.append(story)
                story = []
                continue
            attrs = line.split('\t')
            assert len(attrs) == 1 or len(attrs) == 2
            user = attrs[0].split(' ')
            assert user[0].isdigit()
            user = user[1:]
            bot = None
            if len(attrs) == 2:
                bot = attrs[1].split(' ')
            story.append((user, bot))
    return stories

def time_feature(n):
    return '<' + str(n) + '>'

def convert_story(story):
    ret_stories = []
    for idx, pair in enumerate(story):
        user, bot = pair
        if not bot:
            continue
        ret_story = []
        for i in range(idx):
            u, b = story[i]
            ret_story.append(
                [time_feature(len(ret_story) + 1)] 
#                 + [w.lower() if w != '<SILENCE>' else w for w in u]
                + u
                + ['<USER>']
            )
            if b:
                ret_story.append(
                    [time_feature(len(ret_story) + 1)] 
#                     + [w.lower() if w != '<SILENCE>' else w for w in b]
                    + b
                    + ['<MODEL>']
                )
        ret_stories.append((ret_story, user, bot))
    return ret_stories

def convert2babi(stories):
    babi_stories = []
    for story in stories:
        babi_stories.extend(convert_story(story))
    return babi_stories

def output_babi_story(story):
    facts, question, answer = story
    out_str = []
    for i, fact in enumerate(facts):
        s = ' '.join([str(i + 1)] + fact) + ' .'
        out_str.append(s)
    s = ' '.join([str(len(facts) + 1)] + question) + ' ?'
    s = '\t'.join([s, ' '.join(answer), '1'])
    out_str.append(s)
    return '\n'.join(out_str)

def output_babi_stories(out_file, stories):
    with open(out_file, 'w+') as out_f:
        for story in stories:
            out_str = output_babi_story(story)
            out_f.write(out_str)
            out_f.write('\n')

if __name__ == '__main__':
    dialog_in_dir = './data/dialog-bAbI-tasks/'
    dialog_out_dir = './data/dialog-bAbI-tasks-converted/'
    
    dialog_files = [
        'dialog-babi-task1-API-calls',
        'dialog-babi-task2-API-refine',
        'dialog-babi-task3-options',
        'dialog-babi-task4-phone-address',
        'dialog-babi-task5-full-dialogs',
        'dialog-babi-task6-dstc2'
    ]
    
    extensions = [
        '-trn.txt',
        '-dev.txt',
        '-tst.txt',
        '-tst-OOV.txt'
    ]
    
    for dialog_file in dialog_files[:-1]:
        for extension in extensions:
            filename = dialog_file + extension
            dialog = parse_dialog(os.path.join(dialog_in_dir, filename))
            stories = convert2babi(dialog)
            output_babi_stories(os.path.join(dialog_out_dir, filename), stories)
    
    for dialog_file in dialog_files[-1:]:
        for extension in extensions[:-1]:
            filename = dialog_file + extension
            dialog = parse_dialog(os.path.join(dialog_in_dir, filename))
            stories = convert2babi(dialog)
            output_babi_stories(os.path.join(dialog_out_dir, filename), stories)
    
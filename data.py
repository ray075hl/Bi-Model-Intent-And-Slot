import numpy as np

# convert an empty 2D list into an empty 1D list
flatten = lambda l: [item for sublist in l for item in sublist]

index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]

train_data = open("dataset/atis.train.w-intent.iob", "r").readlines()
test_data = open("dataset/atis.test.w-intent.iob", "r").readlines()

print('This is the first line of data:')

print(train_data[0])

def data_pipeline(data, length=50):
    '''
    [length] represents the standard size of the sequence to be inputed in the model
    This function will make sure that every line from the data has the same length
    before it is fed in the model
    '''
    # remove the '\n' spaces
    data = [t[:-1] for t in data]
    
    # split the data by white spaces
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]  
    
    # transform every line into a dictionary: [ORIGINAL data, LABELED data, and INTEND]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]
    seq_in, seq_out, intent = list(zip(*data))
    
    sin = []
    sout = []
    
    # iterate through every line of the original seq
    for line in range(len(seq_in)):
        ### A D J U S T   T H E   S I Z E   O F   T H E   O R I G I N A L   S E Q U E N C E ###
        temp = seq_in[line]
        # if the line being read is shorter than 'length', this will apply padding to fill it
        if len(temp) < length:
            # <EOS> = End of Sentence
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        
        # if the line being read is larger than 'length', this will cut it to adjust its size
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)
        
        ### A D J U S T   T H E   S I Z E   O F   T H E   L A B E L E D   S E Q U E N C E ###
        temp = seq_out[line]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)
        data = list(zip(sin, sout, intent))
        
    return data

# transform the data so every sequence has the same size/legth
train_data_ed = data_pipeline(train_data)
test_data_ed = data_pipeline(test_data)

print(train_data_ed[0])

def get_info_from_training_data(data):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    
    # generate word2index
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    # generate index2word
    index2word = {v: k for k, v in word2index.items()}

    # generate tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # generate index2tag
    index2tag = {v: k for k, v in tag2index.items()}

    # generate intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    # generate index2intent
    index2intent = {v: k for k, v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent


word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)

print(len(word2index))
print(len(index2word))
print(len(slot2index))
print(len(intent2index))

def to_index(train, word2index, slot2index, intent2index):
    new_train = []
    for sin, sout, intent in train:
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
    return new_train

index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
index_test = to_index(test_data_ed, word2index, slot2index, intent2index)


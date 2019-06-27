words = []

def convert_int(arr):

    try:
        a = int(arr)
    except:
        return None
    return a

with open('train_dev') as f:
    for line in f.readlines():
        line = line.strip().lower().split()

        for index, item in enumerate(line):
            word = item.split(':')[0]
            if word == '<=>':
                break
            if convert_int(word) is not None:
                words.append('DIGIT' * len(word))
            else:        
                words.append(word)

words_vocab = sorted(set(words))

word_dict = {'UNK': 0, 'PAD': 1}

for i, item in enumerate(words_vocab):
    word_dict[item] = i + 2

slot_dict = {}

with open('vocab.slot') as f:

    for i, line in enumerate(f.readlines()):
        slot_dict[line.strip()] = i


print(slot_dict)

intent_dict = {}

with open('vocab.intent') as f:
    for i, line in enumerate(f.readlines()):
        intent_dict[line.strip()] = i

print(intent_dict)



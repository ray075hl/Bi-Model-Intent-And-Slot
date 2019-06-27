from make_dict import word_dict, intent_dict, slot_dict
from make_dict import convert_int
import config as cfg

def makeindex(filename):
    train_data = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split()
            sample_sentence = []
            sample_slot = []
            for index, item in enumerate(line):
                word = item.split(':')[0] 

                if word == '<=>':
                    real_length = index
                    break
                if convert_int(word) is not None:
                    word =  'DIGIT' * len(word)
                else:
                    pass
                slot = item.split(':')[1]

                if word in word_dict:
                    sample_sentence.append(word_dict[word])
                else:
                    sample_sentence.append(word_dict['UNK'])

                sample_slot.append(slot_dict[slot])

                train_intent = intent_dict[ line[-1].split(';')[0] ]

            while len(sample_sentence) < cfg.max_len:
                sample_sentence.append(word_dict['PAD'])
            while len(sample_slot) < cfg.max_len:
                sample_slot.append(slot_dict['O'])

            train_data.append([sample_sentence, real_length, sample_slot, train_intent])
    return train_data


train_data = makeindex(cfg.train_file)
test_data = makeindex(cfg.test_file)

index2slot_dict = {}
for key in slot_dict:
    index2slot_dict[slot_dict[key]] = key


print('Number of training samples: ', len(train_data))
print('Number of test samples: ', len(test_data))
print('Number of words: ', len(word_dict))
print('Number of intent labels: ', len(intent_dict))
print('Number of slot labels', len(slot_dict))


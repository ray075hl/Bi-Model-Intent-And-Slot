from make_dict import word_dict, intent_dict, slot_dict
max_len = 50


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

                slot = item.split(':')[1]

                if word in word_dict:
                    sample_sentence.append(word_dict[word])
                else:
                    sample_sentence.append(word_dict['UNK'])

                sample_slot.append(slot_dict[slot])

                train_intent = intent_dict[ line[-1].split(';')[0] ]

            while len(sample_sentence) < max_len:
                sample_sentence.append(word_dict['PAD'])
            while len(sample_slot) < max_len:
                sample_slot.append(slot_dict['O'])

            train_data.append([sample_sentence, real_length, sample_slot, train_intent])
    return train_data


train_data = makeindex('train_dev')
test_data = makeindex('test')

print('Number of training samples: ', len(train_data))
print('Number of test samples: ', len(test_data))
print('Scale of words vocab: ', len(word_dict))
print('Number of intent labels: ', len(intent_dict))
print('Number of slot labels', len(slot_dict))


from data_iob import word_dict, intent_dict, slot_dict

# train_sentence = []
# train_slot = []
# train_intent = []
# train_sentence_length = []
train_data = []
max_len = 50
word_pad = word_dict['PAD']
slot_pad = len(slot_dict)

with open('atis-2.train+dev.w-intent.iob') as f:
    for line in f.readlines():
        line = line.strip().lower().split()
        sample_sentence = []
        sample_slot = []
        for i, word in enumerate(line[:-1]):
            if word == 'eos':
                index = i
                real_length = index
                break
            sample_sentence.append(word_dict[word])
        
        for slot in line[index+1:-1]:
            sample_slot.append(slot_dict[slot])

        if '#' in line[-1]:
            train_intent = intent_dict[line[-1].split('#')[0]]
        else:
            train_intent = intent_dict[line[-1]]

        while len(sample_sentence) < max_len:
            sample_sentence.append(word_pad)
        while len(sample_slot) < max_len:
            sample_slot.append(slot_pad)
        train_data.append([sample_sentence, real_length, sample_slot, train_intent])


test_data = []
with open('atis.test.w-intent.iob') as f:
    for line in f.readlines():
        line = line.strip().lower().split()
        sample_sentence = []
        sample_slot = []
        for i, word in enumerate(line[:-1]):
            if word == 'eos':
                index = i
                real_length = index
                break
            if word in word_dict:
                sample_sentence.append(word_dict[word])
            else:
                sample_sentence.append(word_dict['UNK'])

        for slot in line[index+1:-1]:
            sample_slot.append(slot_dict[slot])

        if '#' in line[-1]:
            test_intent = intent_dict[line[-1].split('#')[0]]
        else:
            test_intent = intent_dict[line[-1]]

        while len(sample_sentence) < max_len:
            sample_sentence.append(word_pad)
        while len(sample_slot) < max_len:
            sample_slot.append(slot_pad)

        test_data.append([sample_sentence, real_length, sample_slot, test_intent])
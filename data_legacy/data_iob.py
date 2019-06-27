

intent = []
words = []
slots = []
with open('atis-2.train+dev.w-intent.iob') as f:
    for line in f.readlines():
        line = line.strip().lower().split()
        
        for i, word in enumerate(line[:-1]):
            if word == 'eos':
                index = i
                break
            words.append(word)

        for slot in line[index+1:-1]:
            slots.append(slot)

        if '#' in line[-1]:
            for item in line[-1].split('#'):
                intent.append(item)
        else:
            intent.append(line[-1])

# --------------------------------------------------------------

with open('atis.test.w-intent.iob') as f:
    for line in f.readlines():
        line = line.strip().lower().split()

        for i, word in enumerate(line[:-1]):
            if word == 'eos':
                index = i
                break
            # words.append(word)
        
        for slot in line[index+1:-1]:
            slots.append(slot)

        if '#' in line[-1]:
            for item in line[-1].split('#'):
                intent.append(item)
        else:
            intent.append(line[-1])
      
word_vocab = set(words)
intent_vocab = set(intent)
slots_vocab = set(slots)
print(len(word_vocab))
print(len(intent_vocab))
print(len(slots_vocab))


word_dict = {}
intent_dict = {}
slot_dict = {}
for i, key in enumerate(word_vocab):
    word_dict[key] = i
word_dict['UNK'] = len(word_dict)-1
word_dict['PAD'] = len(word_dict)-1
for i, key in enumerate(intent_vocab):
    intent_dict[key] = i

for i, key in enumerate(slots_vocab):
    slot_dict[key] = i

print(slot_dict)


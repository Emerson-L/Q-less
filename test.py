from nltk.corpus import words

game_words = [w for w in words.words() if w.islower() and len(w) >= 3 and w.isalpha()]
print(f"Loaded {len(game_words)} valid words.")
for i in range(100):
    print(game_words[i])
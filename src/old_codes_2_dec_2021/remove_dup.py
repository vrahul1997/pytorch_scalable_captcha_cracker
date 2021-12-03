h = '$$$$$$22$$$$$00$$$$$$$$$11$$$$11$$$22222$$$$$$66$$$$$$$$$$$$$$$$$$$$$$$$$##'
h = h.replace("#", "$")
print(h)

letter = None
word = []
for i in h:
    if i == "$":
        letter = None
    if i != "$" and letter is None:
        letter = i
        word.append(i)
    if i != "$" and letter is not None:
        pass

print(word)
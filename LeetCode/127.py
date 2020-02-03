"""
Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
"""

def ladderLength(beginWord, endWord, wordList):


    def diff(a,b):
        numdiff = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                numdiff += 1
        return numdiff
    if endWord not in wordList:
        return 0
    # if beginWord in wordList:
    #     wordList.remove(beginWord)

    options = []
    for word in wordList:
        if diff(word,beginWord) == 1:
            if word == endWord:
                return 2
            newList = wordList.copy()
            newList.remove(word)
            ladlength = ladderLength(beginWord=word,endWord=endWord,wordList=newList)
            if ladlength == 0:
                options.append(0)
            else:
                options.append(ladlength+1)

    print(beginWord,'-',endWord)
    print('wordList',wordList)
    print('options',options)
    for i in options:
        if i == 0:
            options.remove(0)
    if options == [] or options == [0]:
        return 0
    if options[-1] == 0:
        options.pop(-1)
    print('after',options)
    return min(options)



# print(ladderLength(beginWord = "teach",endWord = "place",wordList = ["peale","wilts","place","fetch","purer","pooch","peace","poach","berra","teach","rheum","peach"]))
print(ladderLength(beginWord = "hit",endWord = "cog",wordList = ["hot","dot","dog","lot","log","cog"]))

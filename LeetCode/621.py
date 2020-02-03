"""
Given a char array representing tasks CPU need to do. It contains capital letters A to Z where different letters represent different tasks. Tasks could be done without original order. Each task could be done in one interval. For each interval, CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between two same tasks, there must be at least n intervals that CPU are doing different tasks or just be idle.

You need to return the least number of intervals the CPU will take to finish all the given tasks.



Example:

Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
"""
def leastInterval(tasks,n):
    tasks_dic = {}
    for task in tasks:
        if task not in tasks_dic.keys():
            tasks_dic[task] = 1
        else:
            tasks_dic[task] += 1
    tasks_dic = dict(sorted(tasks_dic.items(),key = lambda x:x[1],reverse = True))
    rest = 1
    for i in range(1,len(tasks_dic.values())):
        if list(tasks_dic.values())[i-1] == list(tasks_dic.values())[i]:
            rest += 1
        else:
            break
    min_time = (list(tasks_dic.values())[0] - 1) * (n + 1) + rest
    print('N',len(tasks))
    print('mostv',list(tasks_dic.values())[0])
    print('interval',n)
    print(list(tasks_dic.values()))
    print('rest',rest)
    print('min_time',min_time)
    if min_time < len(tasks):
        return tasks
    else:
        return min_time






print(leastInterval(tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2))

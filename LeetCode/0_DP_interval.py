import numpy as np
class interval:
    def __init__(self,s,f,w,name):
        self.s = s
        self.f = f
        self.w = w
        self.name = name
        self.l = f-s


# i1 = interval(s=8,f=10,w=2,name='8-10')
# i2 = interval(s=9,f=13,w=4,name='9-13')
# i3 = interval(s=11,f=14,w=4,name='11-14')
# i4 = interval(s=8.7,f=16,w=9,name='8.7-16')
# i5 = interval(s=15.3,f=17,w=3,name='15.3-17')
# i6 = interval(s=13.5,f=18,w=9,name='13.5-18')
# i7 = interval(s=16.5,f=19,w=1,name='16.5-19')
#
# reqs = [i3,i2,i1,i7,i5,i6,i4]

reqs = []
for i in range(0,100):
    s = np.random.randint(0,2399)
    f = np.random.randint(s,2400)
    w = np.random.randint(1,11)
    name = str(s) + '-' + str(f)
    req = interval(s,f,w,name)
    reqs.append(req)

sorted_reqs = sorted(reqs,key = lambda x:x.f)
p = []
for i in range(0,len(reqs)):
    p.append(None)

computed_opts = []
for i in range(0,len(reqs)):
    computed_opts.append(0)

computed_opts_iter = []
for i in range(0,len(reqs)):
    computed_opts_iter.append(0)

for i in range(0,len(sorted_reqs)).__reversed__():
    for j in range(0,i).__reversed__():
        if sorted_reqs[j].f <= sorted_reqs[i].s:
            p[i] = j
            break

def compute_opt(i):
    if i == None or i == 0:
        return 0
    elif computed_opts[i] != 0:
        return computed_opts[i]
    else:
        computed_opts[i] = max(compute_opt(i-1),compute_opt(p[i]) + sorted_reqs[i].w)
        return computed_opts[i]

def compute_opt_iter():
    computed_opts_iter[0] == sorted_reqs[0].w
    for i in range(1,len(reqs)):
        if p[i] == None:
            computed_opts_iter[i] = max(sorted_reqs[i].w,computed_opts_iter[i-1])
        else:
            computed_opts_iter[i] = max(sorted_reqs[i].w + computed_opts_iter[p[i]],computed_opts_iter[i-1])

def find_solution(i,solution):
    if i == 0:
        return solution
    else:
        if p[i] == None:
            solution.append(sorted_reqs[i].name)
            return solution
        if computed_opts[i-1] <= sorted_reqs[i].w + computed_opts[p[i]]:
            solution.append(sorted_reqs[i].name)
            return find_solution(p[i],solution)
        else:
            return find_solution(i-1,solution)


def find_solution_iter():
    solution = []
    for i in range(1,len(reqs)).__reversed__():
        if computed_opts[i-1] != computed_opts[i]:
             solution.append(sorted_reqs[i].name)
    return solution

value = compute_opt(len(sorted_reqs)-1)
compute_opt_iter()
print(computed_opts)
print(computed_opts_iter)
solution = find_solution(len(sorted_reqs)-1,[])
print(solution)
solution_iter = find_solution_iter()
print(solution)

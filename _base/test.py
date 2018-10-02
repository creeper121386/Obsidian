def cal(x):
    x.append('new')
    print('in func x =', x)

a = ['old']
cal(a)
print('now a =', a)
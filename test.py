# -*- coding: utf-8 -*
'''

@author: ls
'''

a = 3
def Fuc():
    global a
    print (a)  # 1
    a = a + 1
    
    
if __name__ == "__main__":
    print (a)  # 2
    a = a + 1
    Fuc()
    print (a)  # 3
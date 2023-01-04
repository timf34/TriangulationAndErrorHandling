x = 2

def add_two(y):
    return y + 2

z = add_two(x)
print(x, z)
# This works cool


x = [2]

def add_two_to_list_vals(y):
    new_list = []
    for i in y:
        i += 2
        new_list.append(i)

    return new_list

z = add_two_to_list_vals(x)
print(x, z)
# This works


class Val:
    def __init__(self, x):
        self.x = x


def add_again(_val):
    new_ = []

    for i in _val:
        i.x += 2
        new_.append(i)

    return new_

x = [Val(2)]
z = add_again(x)

print(x[0].x, z[0].x)
# This doesnt work! Why?
# Because the list is a pointer to the object, not the object itself
# So when you change the object, you change the list
# But when you change the list, you change the object
# So you have to change the object, not the list

# This is the same as the above, but we don't change the original object, we create a new one
# This is the same as the first example
class Val:
    def __init__(self, x):
        self.x = x

def add_again(_val):
    new_ = []

    for i in _val:
        new_val = Val(i.x + 2)
        new_.append(new_val)

    return new_

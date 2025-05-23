def pyramid(n):
    count = 1
    print("Pyramid")
    for i in range(n):
        # Print spaces
        for j in range(n-i):
            print(' ', end='')

        # Print stars
        for k in range(count):
            print('*', end='')

        
        print()
        count += 2


n = 5
pyramid(n)
def lower_triangle(n):
    print("\nTriangle")
    for i in range(n+1):
        for j in range(i):
            print('*', end=' ')
        print()


n = 5
lower_triangle(n)
def reverse_triangle(n):
    print("\nReverse Triangle")
    for i in range(n):
        for j in range(n):
            if j >= i:
                print('*', end=' ')
            else:
                print(' ', end=' ')
        print()


n = 5
reverse_triangle(n)

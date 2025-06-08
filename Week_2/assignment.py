class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def add_node(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
    
    def print_list(self):
        current = self.head
        elements = []
        while current is not None:
            elements.append(str(current.data))
            current = current.next
        if elements:
            print(" -> ".join(elements))
        else:
            print("List is empty")
    
    def delete_nth_node(self, n):
        if self.head is None:
            print("Error: List is empty")
            return
        
        if n < 1:
            print("Error: Index must be 1 or greater")
            return
        
        if n == 1:
            self.head = self.head.next
            return
        
        current = self.head
        prev = None
        count = 1
        
        while current is not None and count < n:
            prev = current
            current = current.next
            count += 1
        
        if current is None:
            print(f"Error: Index {n} is out of range")
            return
        
        prev.next = current.next


ll = LinkedList()

ll.print_list()

for num in [10, 20, 30, 40, 50]:
    ll.add_node(num)
ll.print_list()

ll.delete_nth_node(3)
ll.print_list()

ll.delete_nth_node(1)
ll.print_list()

ll.delete_nth_node(5)

empty_ll = LinkedList()
empty_ll.delete_nth_node(1)

ll.print_list()

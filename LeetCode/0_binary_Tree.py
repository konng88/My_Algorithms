class TreeNode:
    def __init__(self,**kwds):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

    def grow(self,node,list,counter):
        if len(list) > 1:
            node.left = TreeNode(list.pop(0))
            counter += 1
            node.right == TreeNode(list.pop(0))
            counter += 1
            brother = node.right
        if len(list) == 1:
            node.left = TreeNode(list.pop(0))
        if len(list) == 0:
            return


    def print_tree(self,node):
        if node == None:
            return
        print(node.val)
        self.print_tree(node.left)
        self.print_tree(self.right)

tree = TreeNode(None)
tree = tree.grow(tree,[3,9,20,'null','null',15,7],0)
print(tree.left.left.val)

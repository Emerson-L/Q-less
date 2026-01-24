import utils
from treelib.tree import Tree

BREAK = '>'
EOW = '$'

def add_nodes(tree, parent_id, data, idstr):
    for key, value in data.items():
        # Create a unique ID for the node to avoid collisions
        node_id = idstr + key
        
        # Add the node to the tree
        tree.create_node(tag=key, identifier=node_id, parent=parent_id)
        
        # If the value is a dictionary, recurse deeper
        if isinstance(value, dict):
            add_nodes(tree, node_id, value, node_id)


def get_gddag(words: list[str]) -> dict:
    root = {}
    for w in words:
        for i in range(1, len(w)+1):
            cur_node = root
            if i == len(w):
                rep = w[::-1] + BREAK + EOW
            else:
                rep = w[0:i][::-1] + BREAK + w[i:len(w)] + EOW
            print(f'{i}: {rep}')
            for l in rep:
                if l not in cur_node:
                    cur_node[l] = {}
                cur_node = cur_node[l]
    return root
                

if __name__ == '__main__':
    # words = utils.load_words()[:3]
    words = ['call', 'ball', 'all']
    gddag = get_gddag(words)
    tree = Tree()
    tree.create_node(tag='Root', identifier='root')

    add_nodes(tree, 'root', gddag, '')
    tree.show()
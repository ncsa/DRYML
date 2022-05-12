# Taking after stackoverflow answer here:
# https://stackoverflow.com/questions/32151776/
#     visualize-tree-in-bash-like-the-output-of-unix-tree

import argparse
import os
import zipfile
import functools
import dryml

branch = '├'
pipe = '|'
end = '└'
dash = '─'


class Tree(object):
    def __init__(self, tag):
        self.tag = tag


class Node(Tree):
    def __init__(self, tag, *nodes):
        super().__init__(tag)
        self.nodes = nodes


class Leaf(Tree):
    pass


def _draw_tree(tree, level, last=False, sup=[]):
    def update(left, i):
        if i < len(left):
            left[i] = '   '
        return left

    print(''.join(functools.reduce(update, sup, ['{}  '.format(pipe)] * level))
          + (end if last else branch) + '{} '.format(dash)
          + str(tree.tag))
    if isinstance(tree, Node):
        level += 1
        for node in tree.nodes[:-1]:
            _draw_tree(node, level, sup=sup)
        _draw_tree(tree.nodes[-1], level, True, [level] + sup)


def draw_tree(trees):
    for tree in trees[:-1]:
        _draw_tree(tree, 0)
    _draw_tree(trees[-1], 0, True, [0])


file_blocklist = [
   'meta_data.pkl',
   'cls_def.dill',
   'dry_args.pkl',
   'dry_kwargs.pkl',
   'dry_mut.pkl',
]


def create_tree_from_dryfile(input_file, tag):
    with dryml.DryObjectFile(input_file) as dry_f:
        obj_def = dry_f.definition()
        content_names = dry_f.file.namelist()
        content_nodes = []
        for name in content_names:
            if name[-4:] == '.dry':
                # this is another dry file.
                with dry_f.file.open(name, mode='r') as f:
                    with zipfile.ZipFile(f, mode='r') as sub_zip:
                        content_nodes.append(
                            create_tree_from_dryfile(sub_zip, name))
            elif name not in file_blocklist:
                content_nodes.append(Leaf(name))
        if len(content_nodes) > 0:
            return Node(
                f"{tag}: {obj_def.dry_id} "
                "[{dryml.utils.get_class_str(obj_def.cls)}]",
                *content_nodes)
        else:
            return Leaf(
                f"{tag}: {obj_def.dry_id} "
                "[{dryml.utils.get_class_str(obj_def.cls)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Show content of dry files. They may be nested.")
    parser.add_argument(
        "input",
        help="The filepath to the input dry file", type=str)
    args = parser.parse_args()

    file_path = args.input

    if not os.path.exists(file_path):
        print(f"File {file_path} doesn't exist.")

    with zipfile.ZipFile(file_path, mode='r') as root_file:
        full_tree = create_tree_from_dryfile(root_file, 'root')
        draw_tree([full_tree])

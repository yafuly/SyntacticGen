from collections import  defaultdict
import sys
valid_list = ['TOP', 'NP', 'VP', 'PP', 'S']


# Tree
class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        # assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, nocache=False):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

def helper(index, tokens):
    trees = []

    while index < len(tokens) and tokens[index] == "(":
        paren_count = 0
        while tokens[index] == "(":
            index += 1
            paren_count += 1

        label = tokens[index]
        index += 1

        if tokens[index] == "(":
            children, index = helper(index, tokens)
            # print(children, index, label)
            trees.append(InternalTreebankNode(label, children))
        else:
            word = tokens[index]
            index += 1
            # print(label, word)
            trees.append(LeafTreebankNode(label, word))

        while paren_count > 0:
            assert tokens[index] == ")"
            index += 1
            paren_count -= 1

    return trees, index



# level-order traversal
def traverse_level_order(node):

    def _loop_children(node):
        if not isinstance(node, InternalTreebankNode):
            return []

        labels = []
        for child in node.children:
            if isinstance(child, InternalTreebankNode):
                labels.append(f'<{child.label}>')
            else:
                labels.append(child.word)
        return labels


    layer_seqs = defaultdict(list)
    node_list = [node[0]]
    finished = False
    layer_index = 0
    while not finished:
        src_seq = []
        tgt_seq = []
        next_node_list = []
        cons_index = 0
        
        for n in node_list:
            if isinstance(n, InternalTreebankNode):
                src_seq += [f'<{n.label}>']
                for child in n.children:
                    labels = _loop_children(n)
                    next_node_list.append(child)
                tgt_seq += [f'<CONS-0>'] + labels
                cons_index += 1

            else:
                assert isinstance(n, LeafTreebankNode)
                src_seq.append(n.word)
                next_node_list.append(n)

        layer_seqs[layer_index] = (" ".join(src_seq), " ".join(tgt_seq))
        layer_index += 1
        node_list = next_node_list
        finished = True
        for n in next_node_list:
            if isinstance(n, InternalTreebankNode):
                finished = False

    return layer_seqs

def text_norm(text):
    # restore brackets
    out_text = text.replace("-LSB-","[").replace("-RSB-","]").replace("-LRB-","(").replace("-RRB-",")").replace("-LCB-","{").replace("-RCB-","}")
    # normalize unk token
    out_text = out_text.replace("[UNK]","<unk>")
    return out_text

# prune
def prune(tree):
    if tree.children:
        # print(tree.children)
        tmp_list = []
        for child in tree.children:
            if isinstance(child, InternalTreebankNode):
                prune(child)
                if child.label not in valid_list:
                    tmp_list += child.children 
                else:
                    tmp_list.append(child)       
            else:
                tmp_list.append(child)

        tree.children = tmp_list



# prepare training data
srcl = sys.argv[1]
tgtl = sys.argv[2]
text_path = sys.argv[3]
parse_path = sys.argv[4]
out_path = sys.argv[5]


LANGS = [srcl, tgtl]
SRC = text_path
TGT = parse_path
OUT_PATH = out_path

subsets = ['train', 'valid', 'test']
for s in subsets:
    if s in text_path:
        subset = s
src_lines = [l.strip() for l in open(SRC).readlines()]
tgt_merge_lines = [l.strip() for l in open(TGT).readlines()]
out_src_lines = []
out_tgt_lines = []
flags = []
cache = {}

from tqdm import tqdm
for src_text, tgt_merge in tqdm(zip(src_lines, tgt_merge_lines)):
    idx, tgt_text, parse_line = tgt_merge.split("\t")

    parse_tokens = parse_line.replace("(", " ( ").replace(")", " ) ").split()
    trees, _ = helper(0, parse_tokens)
    prune(trees[0])
    sub_paths = traverse_level_order(trees)
    if src_text not in cache:
        flag = True
    else:
        flag = False
    for k,v in sub_paths.items():
        if flag: # remove duplicate source
            # _out_tgt = text_norm(v[0]) + " <sep> " + text_norm(v[1])
            _out_src = src_text + " <sep> " + text_norm(v[0]) 
            out_tgt_lines.append(text_norm(v[1]))
            out_src_lines.append(_out_src)
    cache[src_text] = ""

import os
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)
with open(f'{OUT_PATH}/{subset}.{LANGS[0]}', 'w') as f:
    f.write("\n".join(out_src_lines)+"\n")
with open(f'{OUT_PATH}/{subset}.{LANGS[1]}', 'w') as f:
    f.write("\n".join(out_tgt_lines)+"\n")
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


from collections import  defaultdict

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
        # print(next_node_list)
        # print([isinstance(e, InternalTreebankNode) for e in next_node_list])
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

valid_list = ['TOP', 'NP', 'VP', 'PP', 'S']


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


def f1_score(true_entities, pred_entities):
    """Compute the F1 score."""
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def find_span(tree, index, end, span):
        if tree.children:
            for child in tree.children:
                if isinstance(child, InternalTreebankNode):
                    # print(child.label, index)
                    start = index
                    index, end, _ = find_span(child, index, end, span, )
                    span.append(str(child.label) + " " + str(start) + " " + str(end-1))
                else:
                    # print(child.word, child.tag, index)
                    # print(child.tag,index)
                    index += 1
                    end += 1
        return index, end, span

def text_norm(text, reverse=False):
    if not reverse:
        # restore brackets
        out_text = text.replace("-LSB-","[").replace("-RSB-","]").replace("-LRB-","(").replace("-RRB-",")").replace("-LCB-","{").replace("-RCB-","}")
        # normalize unk token
        out_text = out_text.replace("[UNK]","<unk>")
    else:
        # protect brackets
        out_text = text.replace("[","-LSB-").replace("]","-RSB-").replace("(","-LRB-").replace(")","-RRB-").replace("{","-LCB-").replace("}","-RCB-")
    return out_text
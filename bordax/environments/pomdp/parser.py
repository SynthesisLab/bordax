import lark

from bordax.environments.pomdp.utils import POMDP

# This parses POMDP files. The formal specification is here:
#
# https://www.pomdp.org/code/pomdp-file-grammar.html

parser = lark.Lark(
    r"""
    pomdp_file      : preamble start? param_list

    // First part: preamble
    preamble        : param_type*
    ?param_type     : discount_param | values_param | state_param | action_param | obs_param

    discount_param  : "discount" ":" discount_tail
    ?discount_tail  : float

    values_param    : "values" ":" values_tail
    ?values_tail    : "reward"      -> reward
                    | "cost"        -> cost

    state_param     : "states" ":" state_tail
    ?state_tail     : int | id_list

    action_param    : "actions" ":" action_tail
    ?action_tail    : int | id_list

    obs_param       : "observations" ":" obs_param_tail
    ?obs_param_tail : int | id_list

    // Second part: start
    start           : "start" ":" u_matrix                   -> start_matrix
                    | "start" ":" string                     -> start_state
                    | "start" "include" ":" start_list -> include
                    | "start" "exclude" ":" start_list -> exclude
    start_list      : id+

    // Third part: param_list
    param_list      : param_spec*
    ?param_spec     : trans_prob_spec | obs_prob_spec | reward_prob_spec

    ?trans_prob_spec: "T" ":" trans_spec_tail
    trans_spec_tail : id ":" id ":" id prob -> trans_entry
                    | id ":" id u_matrix    -> trans_row
                    | id ui_matrix          -> trans_matrix

    ?obs_prob_spec  : "O" ":" obs_spec_tail
    obs_spec_tail   : id ":" id ":" id prob -> obs_entry
                    | id ":" id u_matrix    -> obs_row
                    | id u_matrix           -> obs_matrix

    ?reward_prob_spec: "R" ":" reward_spec_tail
    reward_spec_tail: id ":" id ":" id ":" id number -> reward_entry
                    | id ":" id ":" id u_matrix    -> reward_row
                    | id u_matrix           -> reward_matrix

    ?ui_matrix      : "uniform"    -> uniform
                    | "identity"   -> identity
                    | prob_matrix

    ?u_matrix       : "uniform"    -> uniform
                    | "reset"      -> reset
                    | prob_matrix

    prob_matrix     : prob+
    ?id             : int
                    | string
                    | "*"    -> asterisk
    id_list         : string+
    prob            : signed_number
    ?int            : INTTOK
    string          : STRINGTOK
    ?float          : FLOATTOK

    ?number: signed_number

    ?signed_number: SIGNED_FLOATTOK -> float
                | SIGNED_INTTOK -> int

    SIGNED_FLOATTOK: /[+-]?([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?/
    SIGNED_INTTOK: /[+-]?[0-9]+/

    INTTOK: /0|[1-9][0-9]*'/
    FLOATTOK: /([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?/
    STRINGTOK: /[a-zA-Z]([a-zA-Z0-9]|[\_\-])*/
    COMMENT: "#" /[^\n]*/ "\n"

    %import common.WS
    %ignore WS
    %ignore COMMENT
    """,
    start="pomdp_file",
    parser="lalr",
)


class TreeSimplifier(lark.Transformer):
    def string(self, s):
        (s,) = s
        return s[:]  # copies the string

    def prob(self, n):
        (n,) = n
        n = float(n)
        assert 0.0 <= n <= 1.0
        return n

    def int(self, n):
        (n,) = n
        return int(n)

    def float(self, n):
        (n,) = n
        return float(n)

    start_list = list
    id_list = list
    prob_matrix = list


class TreeToSets(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def discount_param(self, tree):
        self.pomdp.setDiscount(tree.children[0])

    def values_param(self, tree):
        self.pomdp.setValues(tree.children[0])

    def state_param(self, tree):
        self.pomdp.setStates(tree.children[0])

    def action_param(self, tree):
        self.pomdp.setActions(tree.children[0])

    def obs_param(self, tree):
        self.pomdp.setObs(tree.children[0])


class TreeToProbs(lark.Visitor):
    def __init__(self, pomdp):
        self.pomdp = pomdp

    def start_matrix(self, tree):
        child = tree.children[0]
        assert isinstance(child, lark.Tree)
        if child.data == "uniform":
            self.pomdp.setUniformStart()
        else:
            assert False

    def include(self, tree):
        child = tree.children[0]
        self.pomdp.setUniformStart(inc=child)

    def exclude(self, tree):
        child = tree.children[0]
        self.pomdp.setUniformStart(exc=child)

    def trans_matrix(self, tree):
        (action, matrix) = tree.children
        if isinstance(action, lark.Tree) and action.data == "asterisk":
            action = None
        if isinstance(matrix, lark.Tree):
            if matrix.data == "uniform":
                self.pomdp.addUniformTrans(act=action)
            elif matrix.data == "identity":
                self.pomdp.addIdentityTrans(act=action)
            else:
                assert False
        else:
            self.pomdp.addTrans(matrix, act=action)

    def obs_matrix(self, tree):
        (action, matrix) = tree.children
        if isinstance(action, lark.Tree) and action.data == "asterisk":
            action = None
        if isinstance(matrix, lark.Tree):
            if matrix.data == "uniform":
                self.pomdp.addUniformObs(act=action)
            else:
                assert False
        else:
            self.pomdp.addObs(matrix, act=action)

    def reward_entry(self, tree):
        # the rewards can depend on the action, initial state, final state and observations

        (action, state, next_state, observation, reward) = tree.children
        if isinstance(action, lark.Tree) and action.data == "asterisk":
            action = None
        if isinstance(state, lark.Tree) and state.data == "asterisk":
            state = None
        if isinstance(next_state, lark.Tree) and next_state.data == "asterisk":
            next_state = None
        if isinstance(observation, lark.Tree) and observation.data == "asterisk":
            observation = None
        self.pomdp.addReward(
            action=action,
            start_state=state,
            next_state=next_state,
            observation=observation,
            reward=reward,
        )

    def reward_row(self, tree):
        # TO DO
        assert False

    def reward_matrix(self, tree):
        # TO DO
        assert False


def parse(instr):
    cst = parser.parse(instr)
    ast = TreeSimplifier().transform(cst)
    pomdp = POMDP()
    TreeToSets(pomdp).visit(ast)
    TreeToProbs(pomdp).visit(ast)
    if pomdp.start == {}:
        pomdp.setUniformStart()
    return pomdp

import random

from stan_ast import *

def python_to_ast(value):
    if isinstance(value, int):
        return LitInt(value=value)

    elif isinstance(value, float):
            return LitReal(value=value, text=None)
    
    elif value is None:
        return None
        
    elif isinstance(value, PyExpr):
        return value.ast_node
    
    else:
        return value


class PyExpr:
    def __init__(self, model, ast_node, id=random.randint(0, 2**63)):
        self.model = model

        self.id = model.counter
        model.counter += 1

        self.ast_node = python_to_ast(ast_node)

    def __getitem__(self, index):
        # Wrap single items in a list
        if isinstance(index, list):
            indices = index
        elif isinstance(index, tuple):
            indices = list(index)
        else:
            indices = [index]

        ast_indices = [python_to_ast(i) for i in indices]

        inner_ast_node = Subscripts(self.ast_node, ast_indices)
    
        py_expr = PyExpr(self.model, inner_ast_node)
    
        return py_expr
    
    def __repr__(self):
        return "<PyExpr[{id}] {expr}>".format(
            expr=self.ast_node,
            id=self.id
        )
    
    def __xor__(self, other):
        sample_statement = Sample(self.ast_node, other.ast_node)

        self.model.statements.append(sample_statement)

        return PyExpr(self.model, sample_statement)

    def __lshift__(self, other):
        assignment_statement = Assignment(self.ast_node, other.ast_node)

        self.model.statements.append(assignment_statement)

        return PyExpr(self.model, assignment_statement)
    
    def __add__(self, other):
        return PyExpr(
            self.model,
            BinOp('+', self.ast_node, other.ast_node)
        )
    
    def __mul__(self, other):
        return PyExpr(
            self.model,
            BinOp('*', self.ast_node, other.ast_node)
        )
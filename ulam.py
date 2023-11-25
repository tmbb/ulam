from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import hashlib
import os
from cmdstanpy import CmdStanModel
import arviz
    
from io import StringIO
import pprint

from py_expr import PyExpr
from stan_ast import *
from stan_functions_library import StanFunctionsLibrary

INDENT_DELTA = 2

LPDF_LIKE_SUFFIXES = [
    '_lpdf', '_lupdf', '_lpmf', '_lupmf', '_cdf', '_lcdf', '_lccdf'
]

@dataclass
class StanProgram:
    """Representation of a full StanProgram"""
    filename: Optional[str] = field(default_factory=lambda: None)
    functions: List = field(default_factory=lambda: [])
    data: List = field(default_factory=lambda: [])
    transformed_data: List = field(default_factory=lambda: [])
    parameters: List = field(default_factory=lambda: [])
    transformed_parameters: List = field(default_factory=lambda: [])
    model: List = field(default_factory=lambda: [])
    generated_quantities: List = field(default_factory=lambda: [])
    indent_level: int = field(default_factory=lambda: 2)

    def _render_block(self, block_name, statements):
        if not (statements is None or (isinstance(statements, Statements) and statements.statements == [])):

            if isinstance(statements, Statements):
                stmts = statements
            else:
                stmts = Statements(statements=statements)

            whitespace = " " * self.indent_level

            return "{block_name} {{\n{whitespace}{code}\n}}".format(
                whitespace=whitespace,
                block_name=block_name,
                code=serialize_ast(stmts, indent_level=self.indent_level)
            )
        
        else:
            return None
        
    def _merge_field(self, left, right):
        if isinstance(left, Statements):
            stmts_left = left
        elif isinstance(left, list):
            stmts_left = Statements(statements=left)
        else:
            raise Exception("argument must be either a Statements() or a list")

        if isinstance(right, Statements):
            stmts_right = right
        elif isinstance(right, list):
            stmts_right = Statements(statements=right)
        else:
            raise Exception("argument must be either a Statements() or a list")

        new_stmts = Statements(
            statements=stmts_left.statements + stmts_right.statements
        )

        return new_stmts
    
    def merge(self, new_stan_program):
        self.functions = self._merge_field(self.functions, new_stan_program.functions)
        self.data = self._merge_field(self.data, new_stan_program.data)
        self.transformed_data = self._merge_field(self.transformed_data, new_stan_program.transformed_data)
        self.parameters = self._merge_field(self.parameters, new_stan_program.parameters)
        self.transformed_parameters = self._merge_field(self.transformed_parameters, new_stan_program.transformed_parameters)
        self.model = self._merge_field(self.model, new_stan_program.model)
        self.generated_quantities = self._merge_field(self.generated_quantities, new_stan_program.generated_quantities)

        
    def merge_except_for_data_block(self, new_stan_program):
        self.functions = self._merge_field(self.functions, new_stan_program.functions)
        self.transformed_data = self._merge_field(self.transformed_data, new_stan_program.transformed_data)
        self.parameters = self._merge_field(self.parameters, new_stan_program.parameters)
        self.transformed_parameters = self._merge_field(self.transformed_parameters, new_stan_program.transformed_parameters)
        self.model = self._merge_field(self.model, new_stan_program.model)
        self.generated_quantities = self._merge_field(self.generated_quantities, new_stan_program.generated_quantities)


    def prewalk_blocks(self, acc, function):
        new_acc = acc
        (self.functions, new_acc) = prewalk(self.functions, new_acc)
        (self.data, new_acc) = prewalk(self.data, new_acc)
        (self.transformed_data, new_acc) = prewalk(self.transformed_data, new_acc)
        (self.parameters, new_acc) = prewalk(self.parameters, new_acc)
        (self.transformed_parameters, new_acc) = prewalk(self.transformed_parameters, new_acc)
        (self.model, new_acc) = prewalk(self.model, new_acc)
        (self.generated_quantities, new_acc) = prewalk(self.generated_quantities, new_acc)
        
        return new_acc


    def serialize(self):
        rendered_blocks = [
            self._render_block('functions', self.functions),
            self._render_block('data', self.data),
            self._render_block('transformed data', self.transformed_data),
            self._render_block('parameters', self.parameters),
            self._render_block('transformed parameters', self.transformed_parameters),
            self._render_block('model', self.model),
            self._render_block('generated quantities', self.generated_quantities),
        ]

        non_null_rendered_blocks = [b for b in rendered_blocks if b]

        return "\n\n".join(non_null_rendered_blocks)

    def show(self):
        print(self.serialize())


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



def build_for_loops(statements):
    parsed_statements = []
    stack = []

    for stmt in statements:
        if isinstance(stmt, EnterForLoop):
            new_top_of_stack = (stmt, [])
            stack.append(new_top_of_stack)
        
        elif isinstance(stmt, ExitForLoop):
            (enter_for_loop, for_loop_statements) = stack.pop()
            # Make sure the start and end of the loop match
            assert stmt.for_loop_id == enter_for_loop.for_loop_id

            new_statement = ForLoop(
                enter_for_loop.variable.ast_node,
                enter_for_loop.lower.ast_node,
                enter_for_loop.upper.ast_node,
                Statements(for_loop_statements),
                block=enter_for_loop.block
            )

            if stack == []:
                parsed_statements.append(new_statement)

            else:
                (enter_for_loop, for_loop_statements) = stack.pop()
                # Append the new statement to the opern for loop
                for_loop_statements.append(new_statement)
                # Replace the head of the stack
                stack.append((enter_for_loop, for_loop_statements))

        elif stack == []:
            parsed_statements.append(stmt)

        else:
            (enter_for_loop, for_loop_statements) = stack.pop()
            # Append the new statement to the opern for loop
            for_loop_statements.append(stmt)
            # Replace the head of the stack
            stack.append((enter_for_loop, for_loop_statements))

    # The stack should be empty now that all tokens have been processed
    # assert stack == []

    # Return the parsed statements
    return parsed_statements
    

def serialize_constraints(constraints, indent_level=0):
    non_null_constraints = [(c[0], c[1]) for c in constraints if c[1] is not None]

    serialized_non_null_constraints = [
        "{}={}".format(c[0], serialize_ast(c[1], indent_level=indent_level))
        for c in non_null_constraints
    ]

    if non_null_constraints:
        constraints_str = "<" + ", ".join(serialized_non_null_constraints) + ">"
    else:
        constraints_str = ""

    return constraints_str


def serialize_singleton_type(typ):
    if isinstance(typ, TypeInt):
        type_name = 'int'
    elif isinstance(typ, TypeReal):
        type_name = 'real'
    else:
        raise Exception("Invalid singleton type {}".format(type_name))
    
    constraints_str = serialize_constraints([
        ('lower', typ.lower),
        ('upper', typ.upper)
    ])

    return "{type_name}{constraints}".format(
        type_name=type_name,
        constraints=constraints_str
    )

def serialize_array(array):
    dimensions = ", ".join([
        serialize_ast(dim) for dim in array.dimensions
    ])

    return "array[{dimensions}] {type}".format(
        dimensions=dimensions,
        type=serialize_ast(array.type)
    )

def serialize_vector_type(vector_type):
    if isinstance(vector_type, TypeVector):
        type_name = 'vector'
    elif isinstance(vector_type, TypeUnitVector):
        type_name = 'unit_vector'
    elif isinstance(vector_type, TypeSimplex):
        type_name = 'simplex'
    elif isinstance(vector_type, TypeOrderedVector):
        type_name = 'ordered_vector'
    else:
        raise Exception("Invalid vector type {}".format(vector_type))
    
    constraints_str = serialize_constraints([
        ('lower', vector_type.lower),
        ('upper', vector_type.upper),
        ('multiplier', vector_type.multiplier),
        ('offset', vector_type.offset),
    ])

    return "{type_name}{constraints}[{dimension}]".format(
        type_name=type_name,
        constraints=constraints_str,
        dimension=serialize_ast(vector_type.dimension),
    )

def serialize_ast(ast_node, indent_level=0):
    if isinstance(ast_node, Variable):
        return ast_node.name
    
    elif isinstance(ast_node, Comment):
        whitespace = " " * indent_level
        lines = ast_node.text.split("\n")
        return "\n{}".format(whitespace).join([
            "// ".format(whitespace) + line for line in lines
        ]) + ("\n" * ast_node.new_lines)

    elif isinstance(ast_node, Statements):
        whitespace = " " * indent_level
        return "\n{}".format(whitespace).join([
            serialize_ast(stmt, indent_level=indent_level) for stmt in ast_node.statements
        ]) 

    elif isinstance(ast_node, VariableDeclaration):
        return "{type} {variable};".format(
            type=serialize_ast(ast_node.type),
            variable=serialize_ast(ast_node.variable)
        )
    
    elif isinstance(ast_node, TypeArray):
        return serialize_array(ast_node)

    elif isinstance(ast_node, (TypeInt, TypeReal)):
        return serialize_singleton_type(ast_node)

    elif isinstance(ast_node, (TypeVector, TypeUnitVector, TypeSimplex, TypeOrderedVector)):
        return serialize_vector_type(ast_node)

    elif isinstance(ast_node, Sample):
        return "{} ~ {};".format(
            serialize_ast(ast_node.left, indent_level),
            serialize_ast(ast_node.right, indent_level)
        )

    elif isinstance(ast_node, Assignment):
        return "{} = {};".format(
            serialize_ast(ast_node.left, indent_level),
            serialize_ast(ast_node.right, indent_level)
        )

    elif isinstance(ast_node, FunctionCall):
        if any(ast_node.function.endswith(suffix) for suffix in LPDF_LIKE_SUFFIXES):
            arg1 = serialize_ast(ast_node.arguments[0])
            arg2 = serialize_ast(ast_node.arguments[1])
            other_args = ast_node.arguments[2:]

            serialized_arguments = ", ".join(
                ["{} | {}".format(arg1, arg2)] + [
                    serialize_ast(arg, indent_level) for arg in other_args
                ]
            )

        else:
            serialized_arguments = ", ".join(
                (serialize_ast(arg, indent_level) for arg in ast_node.arguments)
            )

        return "{}({})".format(ast_node.function, serialized_arguments)

    elif isinstance(ast_node, Subscripts):
        serialized_indices = ", ".join(
            (serialize_ast(index, indent_level) for index in ast_node.indices)
        )

        serialized_expression = serialize_ast(ast_node.expression, indent_level)
        return "{}[{}]".format(serialized_expression, serialized_indices)

    elif isinstance(ast_node, LitInt):
        return str(ast_node.value)

    elif isinstance(ast_node, LitReal):
        if ast_node.text is None:
            return str(ast_node.value)
        else:
            return ast_node.text

    elif isinstance(ast_node, If):
        return "if ({}) {{\n{}{}\n{}}} else {{\n{}{}\n{}}}".format(
            serialize_ast(ast_node.condition),
            " " * (indent_level + INDENT_DELTA),
            serialize_ast(ast_node.then, indent_level=indent_level + INDENT_DELTA),
            " " * indent_level,
            " " * (indent_level + INDENT_DELTA),
            serialize_ast(ast_node.otherwise, indent_level=indent_level + INDENT_DELTA),
            " " * indent_level
        )
    
    elif isinstance(ast_node, IncrementTarget):
        return "target += {value};".format(ast_node.value)

    elif isinstance(ast_node, ForLoop):
        whitespace = " " * indent_level
        whitespace_further_indented = " " * (indent_level + INDENT_DELTA)

        serialized_body = whitespace_further_indented + serialize_ast(ast_node.body, indent_level + INDENT_DELTA)

        return "\n{indent1}for ({variable} in {lower}:{upper}) {{\n{body}\n{indent2}}}".format(
            indent1=whitespace,
            variable=serialize_ast(ast_node.variable, indent_level),
            lower=serialize_ast(ast_node.lower, indent_level),
            upper=serialize_ast(ast_node.upper, indent_level),
            body=serialized_body,
            indent2=whitespace
        )
    
    elif isinstance(ast_node, BinOp):
        return "{} {} {}".format(
            serialize_ast(ast_node.left, indent_level),
            ast_node.operator,
            serialize_ast(ast_node.right, indent_level),
        )

    else:
        raise Exception("Invalid ast node: {}".format(ast_node))


def is_childless(ast_node):
    return isinstance(ast_node, (
        LitInt, LitReal, TypeInt, TypeReal,
        TypeArray, TypeVector, TypeSimplex, TypeUnitVector, TypeOrderedVector,
        Comment, Variable, VariableDeclaration
    ))

def map_reduce_prewalk_list(elements, acc, function):
    new_acc = acc
    new_elements = []

    for element in elements:
        (new_element, new_acc) = prewalk(element, new_acc, function)
        new_elements.append(new_element)

    return (new_elements, new_acc)


def prewalk(ast_node, acc, function):
    # Inside the function body we standardize on always shadowing the new_acc
    # variable, so that we never forget to "update" it by using acc instead.
    # Forgetting to update it often leads to nasty bugs.
    new_acc = acc

    if is_childless(ast_node):
        return function(ast_node, new_acc)
    
    elif isinstance(ast_node, ForLoop):
        (new_variable, new_acc) = function(ast_node.variable, new_acc)
        (new_lower, new_acc) = function(ast_node.lower, new_acc)
        (new_upper, new_acc) = function(ast_node.upper, new_acc)
        (new_body, new_acc) = function(ast_node.body, new_acc)

        new_for_loop = ForLoop(
            variable=new_variable,
            lower=new_lower,
            upper=new_upper,
            body=new_body,
            block=ast_node.block,
            location=ast_node.location
        )

        return function(new_for_loop, new_acc)


    elif isinstance(ast_node, FunctionCall):
        # Transform the arguments
        (new_arguments, new_acc) = map_reduce_prewalk_list(ast_node.arguments, new_acc, function)
        # Build a new function call with the transformed arguments     
        new_function_call = FunctionCall(function=ast_node.function, arguments=new_arguments)

        # Now, apply the function to the intermeditate function call,
        # in which the function arguments have already been transformed.
        return function(new_function_call, new_acc)
    
    elif isinstance(ast_node, IncrementTarget):
        # Transform the arguments
        (new_value, new_acc) = prewalk(ast_node.value, new_acc, function)
        # Build a new function call with the transformed arguments     
        new_increment_target = IncrementTarget(value=new_value)

        # Now, apply the function to the intermeditate function call,
        # in which the function arguments have already been transformed.
        return function(new_increment_target, new_acc)

    elif isinstance(ast_node, Subscripts):
        (new_indices, new_acc) = map_reduce_prewalk_list(ast_node.indices, new_acc, function)
        (new_expression, new_acc) = prewalk(ast_node.expression, new_acc, function)
        # Build a new subcriped object
        new_subscripts =  Subscripts(expression=new_expression, indices=new_indices)

        # Apply the function to the transformed object
        return function(new_subscripts, new_acc)

    elif isinstance(ast_node, Sample):
        # Transform the left and right hand side (always keeping the accumulator)
        (new_left, new_acc) = prewalk(ast_node.left, new_acc, function)
        (new_right, new_acc) = prewalk(ast_node.right, new_acc, function)
        # Build a new sampling statement
        new_sample = Sample(left=new_left, right=new_right)

        return function(new_sample, acc)

    elif isinstance(ast_node, If):
        # Transform the condition and the branches (always keeping the accumulator)
        (new_condition, new_acc) = prewalk(ast_node.condition, acc, function)
        (new_then, new_acc) = prewalk(ast_node.then, acc, function)
        (new_otherwise, new_acc) = prewalk(ast_node.otherwise, acc, function)        
        # Build a new if statement
        new_if = If(condition=new_condition, then=new_then, otherwise=new_otherwise)

        return function(new_if, new_acc)

    elif isinstance(ast_node, BinOp):
        (new_left, new_acc) = prewalk(ast_node.left, new_acc, function)
        (new_right, new_acc) = prewalk(ast_node.right, new_acc, function)
        # Build new BinOp
        new_bin_op = BinOp(operator=ast_node.operator, left=new_left, right=new_right)
        
        return function(new_bin_op, new_acc)

    elif isinstance(ast_node, list):
        (new_arguments, new_acc) = map_reduce_prewalk_list(ast_node, new_acc, function)
        # Transform the new list
        return function(new_arguments, new_acc)

    else:
        raise Exception("Invalid ast node: {}".format(ast_node))

def handle_missing_data(expression, subscripts):
    variable = subscripts.expression
    indices = subscripts.indices

    var__is_missing = Variable("{}__is_missing".format(variable.name))
    var__missing_data_index = Variable("{}__missing_data_index".format(variable.name))
    var__missing = Variable("{}__missing".format(variable.name))
    var__not_missing = Variable("{}__not_missing".format(variable.name))

    def replace_variable_if_datapoint_is_missing(ast_node, acc):
        if isinstance(ast_node, Subscripts) and ast_node.expression == variable:
            new_ast_node = Subscripts(
                expression=var__missing,
                indices=[
                    Subscripts(
                        expression=var__missing_data_index,
                        indices=ast_node.indices
                    )
                ]    
            )

            return (new_ast_node, acc)

        else:
            return (ast_node, acc)

    def replace_variable_if_datapoint_is_not_missing(ast_node, acc):
        if isinstance(ast_node, Subscripts) and ast_node.expression == variable:
            new_ast_node = Subscripts(
                expression=var__not_missing,
                indices=[
                    Subscripts(
                        expression=var__missing_data_index,
                        indices=ast_node.indices
                    )
                ]    
            )

            return (new_ast_node, acc)

        else:
            return (ast_node, acc)

    (branch_if_datapoint_is_missing, _acc) = prewalk(expression, None, replace_variable_if_datapoint_is_missing)
    (branch_if_datapoint_is_not_missing, _acc) = prewalk(expression, None, replace_variable_if_datapoint_is_not_missing)

    condition = Subscripts(
        expression=var__is_missing,
        indices=indices
    )

    if_statement = If(
        condition=condition,
        then=branch_if_datapoint_is_missing,
        otherwise=branch_if_datapoint_is_not_missing
    )

    return if_statement

def decompose_missing_variable_declaration(variable_declaration):
    variable = variable_declaration.variable
    var_type = variable_declaration.type

    if isinstance(var_type, Vector):
        vector_lower = var_type.lower
        vector_upper = var_type.upper
        vector_dimension = var_type.dimension

    else:
        raise Exception("Invalid type for variable with missing data")

    N_missing = Variable(f"{vector_dimension.name}__{variable.name}__missing")
    N_not_missing = Variable(f"{vector_dimension.name}__{variable.name}__not_missing")

    comment_header = Comment(text=(
        "------------------------------------------------------------\n" +
        "Variable {} modified to hold missing data by Ulam.\n".format(variable.name) +
        " START of code generated for missing data\n" +
        "------------------------------------------------------------"
    ), new_lines=0)

    comment_footer = Comment(text=(
        "------------------------------------------------------------\n" +
        " END of code generated for missing data\n" +
        "------------------------------------------------------------"
    ), new_lines=1)

    decl_N_missing = VariableDeclaration(
        variable=N_missing,
        group='data',
        type=TypeInt(lower=LitInt(0), upper=vector_dimension),
        contains_missing_data=False
    )

    decl_N_not_missing = VariableDeclaration(
        variable=N_not_missing,
        group='data',
        type=TypeInt(lower=LitInt(0), upper=vector_dimension),
        contains_missing_data=False
    )

    decl_is_missing = VariableDeclaration(
        variable=Variable(variable.name + "__is_missing"),
        group=variable_declaration.group,
        type=TypeArray(
            dimensions=[vector_dimension],
            type=TypeInt(lower=LitInt(0), upper=LitInt(1))
        ),
        contains_missing_data=False
    )

    decl_missing_data_index = VariableDeclaration(
        variable=Variable(variable.name + "__missing_data_index"),
        group=variable_declaration.group,
        type=TypeArray(
            dimensions=[vector_dimension],
            type=TypeInt(lower=LitInt(0), upper=vector_dimension)
        ),
        contains_missing_data=False
    )

    decl_missing = VariableDeclaration(
        variable=Variable(variable.name + "__missing"),
        group=variable_declaration.group,
        type=TypeVector(
            dimension=N_missing,
            lower=vector_lower,
            upper=vector_upper,
            offset=None,
            multiplier=None
        ),
        contains_missing_data=False
    )

    decl_not_missing = VariableDeclaration(
        variable=Variable(variable.name + "__not_missing"),
        group=variable_declaration.group,
        type=TypeVector(
            dimension=N_not_missing,
            lower=vector_lower,
            upper=vector_upper,
            offset=None,
            multiplier=None
        ),
        contains_missing_data=False
    )

    data_statements = Statements(
        statements=[
            comment_header,
            decl_N_missing,
            decl_N_not_missing,
            decl_is_missing,
            decl_missing_data_index,
            decl_not_missing,
            comment_footer
        ]
    )

    parameters_statements = Statements(
        statements=[
            comment_header,
            decl_missing,
            comment_footer
        ]
    )

    return StanProgram(
        data=data_statements,
        parameters=parameters_statements
    )



class ForLoopRangeIteratorState(Enum):
    INACTIVE = 0
    ACTIVE = 1
    

class ForLoopRangeIterator:
    def __init__(self, ulam_model, variable, lower, upper, block=None, location=Location()):
        self.variable = variable
        self.lower = lower
        self.upper = upper
        self.for_loop_id = ulam_model.unique_id()
        self.location = location
        self.ulam_model = ulam_model
        self.block = block

    def __iter__(self):
        self.state = ForLoopRangeIteratorState.INACTIVE
        return self
    
    def __next__(self):
        if self.state == ForLoopRangeIteratorState.INACTIVE:
            self.ulam_model._enter_for_loop(
                self.for_loop_id,
                self.variable,
                self.lower,
                self.upper,
                self.block
            )

            self.state = ForLoopRangeIteratorState.ACTIVE
            return self.variable
        
        elif self.state == ForLoopRangeIteratorState.ACTIVE:
            self.ulam_model._exit_for_loop(self.for_loop_id)
            raise StopIteration()


class StanAstPrettyPrinter(pprint.PrettyPrinter):
    def format_namedtuple(self, object, stream, indent, allowance, context, level):
        # Code almost equal to _format_dict, see pprint code
        write = stream.write
        write(object.__class__.__name__ + '(')
        object_dict = object._asdict()
        length = len(object_dict)
        if length:
            # We first try to print inline, and if it is too large then we print it on multiple lines
            inline_stream = StringIO()
            self.format_namedtuple_items(object_dict.items(), inline_stream, indent, allowance + 1, context, level, inline=True)
            upper_width = self._width - indent - allowance
            if len(inline_stream.getvalue()) > upper_width:
                self.format_namedtuple_items(object_dict.items(), stream, indent, allowance + 1, context, level, inline=False)
            else:
                stream.write(inline_stream.getvalue())
        write(')')

    def format_namedtuple_items(self, items, stream, indent, allowance, context, level, inline=False):
        # Code almost equal to _format_dict_items, see pprint code
        indent += self._indent_per_level
        write = stream.write
        last_index = len(items) - 1
        if inline:
            delimnl = ', '
        else:
            delimnl = ',\n' + ' ' * indent
            write('\n' + ' ' * indent)
        for i, (key, ent) in enumerate(items):
            last = i == last_index
            write(key + '=')
            self._format(ent, stream, indent + len(key) + 2,
                         allowance if last else 1,
                         context, level)
            if not last:
                write(delimnl)

    def _format(self, object, stream, indent, allowance, context, level):
        # We dynamically add the types of our namedtuple and namedtuple like 
        # classes to the _dispatch object of pprint that maps classes to
        # formatting methods
        # We use a simple criteria (_asdict method) that allows us to use the
        # same formatting on other classes but a more precise one is possible
        if hasattr(object, '_asdict') and type(object).__repr__ not in self._dispatch:
            self._dispatch[type(object).__repr__] = StanAstPrettyPrinter.format_namedtuple
        super()._format(object, stream, indent, allowance, context, level)

def is_data_declaration(stmt):
    return isinstance(stmt, VariableDeclaration) and stmt.group == VariableDeclarationGroup.DATA

def is_parameter_declaration(stmt):
    return isinstance(stmt, VariableDeclaration) and stmt.group == VariableDeclarationGroup.PARAMETER

def is_generated_declaration(stmt):
    return isinstance(stmt, VariableDeclaration) and stmt.group == VariableDeclarationGroup.GENERATED

def is_predicted_variable(stmt):
    return (
        isinstance(stmt, VariableDeclaration) and
        stmt.group == VariableDeclarationGroup.DATA and
        stmt.predict == True
    )


def handle_predicted_variable(var_decl):
    var_name = var_decl.variable.name
    var__hat = Variable(var_name + "__hat")

    stmt = VariableDeclaration(
        variable=var__at,
        group=VariableDeclarationGroup.GENERATED,
        type=var_decl.type
    )

    return StanProgram(generated_quantities=[stmt])


class UlamModel(StanFunctionsLibrary):

    def __init__(self, stan_file=None, logging=False):
        self.stan_file = stan_file
        self.data_pairs = dict()
        self.logging = logging
        self.counter = 1
        self.variables = dict()
        self.variable_names = dict()
        self.functions = dict()
        self.statements = []
        self.program = StanProgram()

    def __enter__(self):
        return self

    def unique_id(self):
        counter = self.counter
        self.counter += 1
        return counter
    
    def __exit__(self, exc_type, exc_value, traceback):
        # Build some actual for loops from the AST like stack instructions
        # (ForLoopEnter and ForLoopExit)
        self.build_for_loops()
        
        # Inspect the generated AST and find the data variables we want to predict
        self._get_and_store_data_pairs()
        # Divide the AST into the proper blocks to create the AST of a working stan program
        self._populate_stan_program()
        # Generate and compile the stan code
        self._setup_cmdstan_model()

    def _get_and_store_data_pairs(self):
        # An iterator which will find all variables we want to predict
        def reducer(ast_node, acc):
            if is_predicted_variable(ast_node):
                acc.append(ast_node.variable.name)
            return (ast_node, acc)
        
        # Gather the 
        (_statements, predicted_variables) = prewalk(self.statements, [], reducer)
        
        self.predicted_variables = predicted_variables

        data_pairs = dict(
            (variable_name, variable_name + '__hat') for variable_name in self.predicted_variables
        )

        self.data_pairs = data_pairs


    def _setup_cmdstan_model(self):
        # Serialize the program we've built
        contents = self.program.serialize()
        
        # Get or generate the stan file name
        if self.stan_file:
            stan_file = self.stan_file
        else:
            suffix = hashlib.sha1(contents.encode()).hexdigest()
            # Create the models directory if it does not exist
            os.makedirs('ulam_models', exist_ok=True)
            stan_file = os.path.join('ulam_models', 'ulam_model_{}.stan').format(suffix)

        with open(stan_file, 'w') as f:
            f.write(contents)

        # Compile the stan model and store it
        self.model = CmdStanModel(stan_file=stan_file)


    
    def sample(self, **kwargs):
        fit = self.model.sample(**kwargs)
        data = kwargs['data']

        posterior_predictive = [
            (variable_name + '__hat') for variable_name in self.predicted_variables
        ]

        observed_data = dict(
            (variable_name, data[variable_name]) for variable_name in self.predicted_variables
        )

        inference_data = arviz.from_cmdstanpy(
            posterior=fit,
            posterior_predictive=posterior_predictive,
            observed_data=observed_data
        )

        return inference_data
    
    def plot_ppc(self, data, *args, **kwargs):
        data_pairs = kwargs.pop('data_pairs', self.data_pairs)
        return arviz.plot_ppc(data, *args, data_pairs=data_pairs, **kwargs)

    def _populate_stan_program(self):
        for stmt in self.statements:
            if is_data_declaration(stmt):
                if stmt.contains_missing_data:
                    # Change this so that it in fact handles missing data
                    sub_program = StanProgram(data=[stmt])
                    self.program.merge(sub_program)
                else:
                    sub_program = StanProgram(data=[stmt])
                    self.program.merge(sub_program)
                
                # if stmt.predict:
                #     sub_program = handle_predicted_variable(stmt)
                #     self.program.merge(sub_program)

            elif is_parameter_declaration(stmt):
                sub_program = StanProgram(parameters=[stmt])
                self.program.merge(sub_program)

            elif is_generated_declaration(stmt):
                sub_program = StanProgram(generated_quantities=[stmt])
                self.program.merge(sub_program)

            elif isinstance(stmt, ForLoop):
                if stmt.block == 'model':
                    sub_program = StanProgram(model=[stmt])
                    self.program.merge(sub_program)
                
                elif stmt.block == 'generated':
                    sub_program = StanProgram(generated_quantities=[stmt])
                    self.program.merge(sub_program)

            elif isinstance(stmt, Sample):
                sub_program = StanProgram(model=[stmt])
                self.program.merge(sub_program)

            elif isinstance(stmt, Assignment):
                raise Exception("Assignment not supported at the top-level")
            
            else:
                raise Exception("Statement not supported at the top-level")
            
        
        # An iterator which will find all variables with missing data
        def reducer(ast_node, acc):
            if is_data_declaration(ast_node) and ast_node.contains_missing_data:
                acc.append(ast_node.variable.name)
            return (ast_node, acc)
        
        (_statements, variables_with_missing_data) = prewalk(self.program.data.statements, [], reducer)
        
        program_after_replacing_missing_data = StanProgram()

        for stmt in self.program.data.statements:
            if is_data_declaration(stmt) and stmt.variable.name in variables_with_missing_data:
                program_after_replacing_missing_data.merge(
                    decompose_missing_variable_declaration(stmt)
                )

            else:
                program_after_replacing_missing_data.merge(
                    StanProgram(data=[stmt])
                )

        program_after_replacing_missing_data.merge_except_for_data_block(self.program)
        # Replace the program wiith the new version
        self.program = program_after_replacing_missing_data

    def _enter_for_loop(self, for_loop_id, variable, lower, upper, block):
        self.statements.append(EnterForLoop(for_loop_id, variable, lower, upper, block))

    def _exit_for_loop(self, for_loop_id):
        self.statements.append(ExitForLoop(for_loop_id))

    def build_for_loops(self):
        statements = build_for_loops(self.statements)
        self.statements = statements

    def _maybe_logging(self, prefix, py_expr):
        if self.logging:
            ast_node = py_expr.ast_node
            location = ast_node.location
            loc = "{}:{}".format(location.file_name, location.line_nr)
            message = "{}: {} - {}".format(prefix, serialize_ast(ast_node), loc)
            print(message)
            return py_expr
        
        else:
            return py_expr

    def put_variable(self, py_expr):
        self.variables[py_expr.id] = py_expr

    def data(self, name, type, missing_data=False, predict=False, log_likelihood=False):
        location = current_python_source_location()
        variable = Variable(name, location=location)

        stmt = VariableDeclaration(
            variable,
            VariableDeclarationGroup.DATA,
            python_to_ast(type),
            contains_missing_data=missing_data,
            predict=predict,
            log_likelihood=log_likelihood,
            location=location
        )

        self.statements.append(stmt)
        
        return PyExpr(self, Variable(name))

    def parameter(self, name, type):
        location = current_python_source_location()
        variable = Variable(name, location=location)

        stmt = VariableDeclaration(
            variable=variable,
            group=VariableDeclarationGroup.PARAMETER,
            type=python_to_ast(type),
            contains_missing_data=False,
            location=location
        )

        self.statements.append(stmt)

        return self._maybe_logging('parameter', PyExpr(self, variable))
    
    def generated(self, name, type):
        location = current_python_source_location()
        variable = Variable(name, location=location)

        stmt = VariableDeclaration(
            variable,
            VariableDeclarationGroup.GENERATED,
            python_to_ast(type),
            location=location
        )

        self.statements.append(stmt)

        return self._maybe_logging('parameter', PyExpr(self, variable))
    
    def range(self, variable_name, lower, upper, block=None):
        location = current_python_source_location()
        variable = Variable(variable_name, location=location)
        py_variable = PyExpr(self, variable)
        py_lower = PyExpr(self, lower)
        py_upper = PyExpr(self, upper)

        self._maybe_logging('loop variable', PyExpr(self, variable))

        return ForLoopRangeIterator(self, py_variable, py_lower, py_upper, block=block, location=location)
        
    def vector(self, size, lower=None, upper=None, offset=None, multiplier=None):
        return TypeVector(
            dimension=python_to_ast(size),
            lower=python_to_ast(lower),
            upper=python_to_ast(upper),
            offset=python_to_ast(offset),
            multiplier=python_to_ast(multiplier)
        )

    def increment_target(self, value):
        statement = IncrementTarget(value)
        
        self.statements.append(statement)

        py_expr = PyExpr(self, statement)
        return py_expr
    
    def integer(self, lower=None, upper=None):
        return TypeInt(lower=python_to_ast(lower), upper=python_to_ast(upper))

    def real(self, lower=None, upper=None):
        return TypeReal(lower=python_to_ast(lower), upper=python_to_ast(upper))

    def _for(self, var, lower, upper, body):
        return PyExpr(self, ForLoop(var, lower, upper, body))
    
    def if_(self, condition, then, otherwise):
        ast_condition = python_to_ast(condition)
        ast_then = python_to_ast(then)
        ast_otherwise = python_to_ast(otherwise)

        return PyExpr(self, If(ast_condition, ast_then, ast_otherwise))
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from inspect import getframeinfo, stack


class VariableDeclarationGroup(Enum):
    DATA = 0
    PARAMETER = 1
    GENERATED = 2


@dataclass
class Location:
    file_name: Optional[str] = None
    line_nr: Optional[int] = None


def current_python_source_location():
    caller = getframeinfo(stack()[2][0])
    return Location(file_name=caller.filename, line_nr=caller.lineno)


class StanAstBuilderToken:
    pass


class StanAstNode:
    pass


@dataclass
class TypeSimplex(StanAstNode):
    dimension: StanAstNode
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    offset: Optional[StanAstNode] = None
    multiplier: Optional[StanAstNode] = None
    location: Optional[Location] = Location()


@dataclass
class TypeUnitVector(StanAstNode):
    dimension: StanAstNode
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    offset: Optional[StanAstNode] = None
    multiplier: Optional[StanAstNode] = None
    location: Optional[Location] = Location()

    
@dataclass
class TypeOrderedVector(StanAstNode):
    dimension: StanAstNode
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    offset: Optional[StanAstNode] = None
    multiplier: Optional[StanAstNode] = None
    location: Optional[Location] = Location()


@dataclass
class TypeOrderedVector(StanAstNode):
    dimension: StanAstNode
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    offset: Optional[StanAstNode] = None
    multiplier: Optional[StanAstNode] = None
    location: Optional[Location] = Location()


@dataclass
class TypeVector(StanAstNode):
    dimension: StanAstNode
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    offset: Optional[StanAstNode] = None
    multiplier: Optional[StanAstNode] = None
    location: Optional[Location] = Location()

@dataclass
class TypeArray(StanAstNode):
    dimensions: StanAstNode
    type: StanAstNode
    location: Optional[Location] = Location()

@dataclass
class TypeInt(StanAstNode):
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    location: Optional[Location] = Location()

    
@dataclass
class TypeReal(StanAstNode):
    lower: Optional[StanAstNode] = None
    upper: Optional[StanAstNode] = None
    location: Optional[Location] = Location()

@dataclass
class LitInt(StanAstNode):
    value: int
    location: Optional[Location] = Location()

@dataclass
class LitReal(StanAstNode):
    value: float
    text: Optional[str] = None
    location: Optional[Location] = Location()

@dataclass
class Statements(StanAstNode):
    statements: List[StanAstNode]
    location: Optional[Location] = Location()

@dataclass
class Comment(StanAstNode):
    text: str
    new_lines: Optional[int] = 0
    location: Optional[Location] = Location()

@dataclass
class Variable(StanAstNode):
    name: str
    location: Optional[Location] = Location()

@dataclass
class VariableDeclaration(StanAstNode):
    variable: Variable
    group: str
    type: StanAstNode
    contains_missing_data: bool = False
    predict: bool = False
    log_likelihood: bool = False
    location: Optional[Location] = Location()

@dataclass
class Sample(StanAstNode):
    left: StanAstNode
    right: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class Assignment(StanAstNode):
    left: StanAstNode
    right: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class FunctionCall(StanAstNode):
    function: StanAstNode
    arguments: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class Subscripts(StanAstNode):
    expression: StanAstNode
    indices: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class IncrementTarget(StanAstNode):
    expression: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class BinOp(StanAstNode):
    operator: str
    left: StanAstNode
    right: StanAstNode
    location: Optional[Location] = Location()


@dataclass
class ForLoop(StanAstNode):
    variable: StanAstNode
    lower: StanAstNode
    upper: StanAstNode
    body: StanAstNode
    block: Optional[str] = None
    location: Optional[Location] = Location()


@dataclass
class If(StanAstNode):
    condition: StanAstNode
    then: StanAstNode
    otherwise: StanAstNode


@dataclass
class EnterForLoop(StanAstBuilderToken):
    for_loop_id: int
    variable: StanAstNode
    lower: StanAstNode
    upper: StanAstNode
    block: Optional[str]


@dataclass
class ExitForLoop(StanAstBuilderToken):
    for_loop_id: int
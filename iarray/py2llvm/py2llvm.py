# Standard Library
import ast
import builtins
import inspect
import math
import operator
import re
from types import FunctionType
import typing

# Requirements
from llvmlite import binding, ir
from llvmlite.ir import Module

# Project
import iarray as ia
from iarray import udf
from . import default
from . import types

# Add UDFJIT to builtins
builtins.UDFJIT = 0


# Plugins
plugins = [default]

MATH_FUNCS = {
    "fabs",
    "fmod",
    "remainder",
    # Exponential functions
    "exp",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    # Power functions
    "sqrt",
    "hypot",
    "pow",
    # Trigonometric functions
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    # Hiperbolic functions
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    # Error and gamma functions
    "erf",
    "lgamma",
    # Nearest ingeger floating-point operations
    "ceil",
    "floor",
    "trunc",
    # Floating-point manipulation functions
    "copysign",
}


class Range:
    def __init__(self, builder, *args):
        start = step = None

        # Unpack
        n = len(args)
        if n == 1:
            (stop,) = args
        elif n == 2:
            start, stop = args
        else:
            start, stop, step = args

        # Defaults
        type_ = stop.type if isinstance(stop, ir.Value) else types.int64
        if start is None:
            start = ir.Constant(type_, 0)
        if step is None:
            step = ir.Constant(type_, 1)

        # Keep IR values
        self.start = types.value_to_ir_value(builder, start)
        self.stop = types.value_to_ir_value(builder, stop)
        self.step = types.value_to_ir_value(builder, step)


def values_to_type(left, right):
    """
    Given two values return their type. If mixing Python and IR values, IR
    wins. If mixing integers and floats, float wins.

    If mixing different lengths the longer one wins (e.g. float and double).
    """
    ltype = types.value_to_type(left)
    rtype = types.value_to_type(right)

    # Both are Python
    if not isinstance(ltype, ir.Type) and not isinstance(rtype, ir.Type):
        if ltype is float or rtype is float:
            return float

        return int

    # At least 1 is IR
    ltype = types.type_to_ir_type(ltype)
    rtype = types.type_to_ir_type(rtype)

    if ltype is types.float64 or rtype is types.float64:
        return types.float64

    if ltype is types.float32 or rtype is types.float32:
        return types.float32

    if ltype is types.int64 or rtype is types.int64:
        return types.int64

    return types.int32


#
# AST
#

LEAFS = {
    ast.Constant,  # 3.8
    ast.Name,
    ast.NameConstant,  # 3.7
    ast.Num,  # 3.7
    # boolop
    ast.And,
    ast.Or,
    # operator
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.MatMult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitXor,
    ast.BitAnd,
    ast.FloorDiv,
    # unaryop
    ast.Invert,
    ast.Not,
    ast.UAdd,
    ast.USub,
    # cmpop
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    # expr_context
    ast.Load,
    ast.Store,
    ast.Del,
    ast.AugLoad,
    ast.AugStore,
    ast.Param,
}


class BaseNodeVisitor:
    """
    The ast.NodeVisitor class traverses the AST and calls user defined
    callbacks when entering a node.

    Here we do the same thing but we've more callbacks:

    - Callback as well when exiting the node
    - Callback as well after traversing an attribute
    - Except leaf nodes, which are called only once (like in ast.NodeVisitor)
    - To find out the callback we use the MRO

    And we pass more information to the callbacks:

    - Pass the parent node to the callback
    - Pass the value of the attribute to the attribute callback
    - Pass the values of all the attributes to the exit callback

    Override this class and define the callbacks you need:

    - def <classname>_enter(node, parent)
    - def <classname>_<attribute>(node, parent, value)
    - def <classname>_exit(node, parent, *args)

    For leaf nodes use:

    - def <classname>_visit(node, parent)

    Call using traverse:

        class NodeVisitor(BaseNodeVisitor):
            ...

        node = ast.parse(source)
        NodeVisitor().traverse(node)
    """

    def __init__(self, debug):
        self.debug = debug
        self.depth = 0

    @classmethod
    def get_fields(cls, node):
        fields = {
            # Skip "decorator_list", and traverse "returns" before "body"
            # ('name', 'args', 'body', 'decorator_list', 'returns')
            ast.FunctionDef: ("name", "args", "returns", "body"),
        }

        return fields.get(type(node), node._fields)

    @classmethod
    def iter_fields(cls, node):
        for field in cls.get_fields(node):
            try:
                yield field, getattr(node, field)
            except AttributeError:
                pass

    def traverse(self, node, parent=None):
        if node.__class__ in LEAFS:
            return self.callback("visit", node, parent)

        # Enter
        # enter callback return False to skip traversing the subtree
        if self.callback("enter", node, parent) is False:
            return None

        self.depth += 1

        # Traverse
        args = []
        for name, field in self.iter_fields(node):
            if isinstance(field, list):
                value = [self.traverse(x, node) for x in field if isinstance(x, ast.AST)]
            elif isinstance(field, ast.AST):
                value = self.traverse(field, node)
            else:
                value = field

            self.callback(name, node, parent, value)
            args.append(value)

        # Exit
        self.depth -= 1
        return self.callback("exit", node, parent, *args)

    def callback(self, event, node, parent, *args):
        for cls in node.__class__.__mro__:
            method = f"{cls.__name__}_{event}"
            cb = getattr(self, method, None)
            if cb is not None:
                break

        # Call
        value = cb(node, parent, *args) if cb is not None else None

        # Debug
        if self.debug > 2:
            name = node.__class__.__name__
            line = None
            if event == "enter":
                line = f"<{name}>"
                if node._fields:
                    attrs = " ".join(f"{k}" for k, _ in ast.iter_fields(node))
                    line = f"<{name} {attrs}>"

                if value is False:
                    line += " SKIP"
            elif event == "exit":
                line = f"</{name}> -> {value}"
            #               if args:
            #                   attrs = ' '.join(repr(x) for x in args)
            #                   line = f'</{name} {attrs}>'
            #               else:
            #                   line = f'</{name}>'
            elif event == "visit":
                if node._fields:
                    attrs = " ".join(f"{k}" for k, _ in ast.iter_fields(node))
                    line = f"<{name} {attrs} />"
                else:
                    line = f"<{name} />"
                if cb is not None:
                    line += f" -> {value}"
            else:
                if cb is not None:
                    attrs = " ".join([repr(x) for x in args])
                    line = f"_{event}({attrs})"

            if line:
                print(self.depth * " " + line)

        return value


class NodeVisitor(BaseNodeVisitor):
    def __init__(self, debug, function):
        super().__init__(debug)
        self.function = function

    def lookup(self, name):
        if name in self.locals:
            return self.locals[name]

        if name in self.root.compiled:
            return self.root.compiled[name]

        if name in self.root.globals:
            return self.root.globals[name]

        libs = ia.udf_registry.libs
        if name in libs:
            return libs[name]

        if name == "UDFJIT":
            return 1

        return getattr(builtins, name)

    def load(self, name):
        try:
            value = self.lookup(name)
        except AttributeError:
            raise ValueError("'name' argument is not found in `inputs` dict")
        if type(value) is ir.AllocaInstr:
            if not isinstance(value.type.pointee, ir.Aggregate):
                return self.builder.load(value)

        return value

    def Module_enter(self, node, parent):
        """
        Module(stmt* body)
        """
        self.root = node

    def FunctionDef_enter(self, node, parent):
        """
        FunctionDef(identifier name, arguments args,
                    stmt* body, expr* decorator_list, expr? returns)
        """
        assert type(parent) is ast.Module, "nested functions not implemented"

        # Initialize function context
        node.locals = {}
        self.locals = node.locals

    def arguments_enter(self, node, parent):
        """
        arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                     arg? kwarg, expr* defaults)
        """
        # We don't parse arguments because arguments are handled in compile
        return False

    def Assign_enter(self, node, parent):
        """
        Assign(expr* targets, expr value)
        Assign(expr* targets, expr value, string? type_comment) # 3.8
        """
        assert len(node.targets) == 1, "Unpacking not supported"

    #
    # Leaf nodes
    #
    def Constant_visit(self, node, parent):
        """
        Constant(constant value, string? kind)
        Pythonr 3.8
        """
        return node.value

    def NameConstant_visit(self, node, parent):
        """
        NameConstant(singleton value)
        Pythonr 3.7
        """
        return node.value

    def Num_visit(self, node, parent):
        """
        Num(object n)
        Pythonr 3.7
        """
        return node.n

    def expr_context_visit(self, node, parent):
        return type(node)

    def Name_visit(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        ctx = type(node.ctx)

        if ctx is ast.Load:
            try:
                return self.lookup(name)
            except AttributeError:
                return None

        elif ctx is ast.Store:
            return name

        raise NotImplementedError(f"unexpected ctx={ctx}")


class InferVisitor(NodeVisitor):
    """
    This optional pass is to infer the return type of the function if not given
    explicitely.
    """

    def Assign_exit(self, node, parent, targets, value, *args):
        target = targets[0]
        if type(target) is str:
            # x =
            self.locals.setdefault(target, value)

    def Return_exit(self, node, parent, value):
        return_type = type(value)

        root = self.root
        if root.return_type is inspect._empty:
            root.return_type = return_type
            return

        assert root.return_type is return_type

    def FunctionDef_exit(self, node, parent, *args):
        root = self.root
        if root.return_type is inspect._empty:
            root.return_type = None


class BlockVisitor(NodeVisitor):
    """
    The algorithm makes 2 passes to the AST. This is the first one, here:

    - We fail early for features we don't support.
    - We populate the AST attaching structure IR objects (module, functions,
      blocks). These will be used in the 2nd pass.
    """

    def FunctionDef_returns(self, node, parent, returns):
        """
        When we reach this point we have all the function signature: arguments
        and return type.
        """
        root = self.root
        ir_signature = root.ir_signature

        # Keep the function in globals so it can be called
        function = root.ir_function
        self.root.compiled[node.name] = function

        # Create the first block of the function, and the associated builder.
        # The first block, named "vars", is where all local variables will be
        # allocated. We will keep it open until we close the function in the
        # 2nd pass.
        block_vars = function.append_basic_block("vars")
        builder = ir.IRBuilder(block_vars)

        # Function start: allocate a local variable for every argument
        args = {}
        for i, param in enumerate(ir_signature.parameters):
            arg = function.args[i]
            assert arg.type is param.type
            ptr = builder.alloca(arg.type, name=param.name)
            builder.store(arg, ptr)
            # Keep Give a name to the arguments, and keep them in local namespace
            args[param.name] = ptr

        # Function preamble
        self.function.preamble(builder, args)

        # Every Python argument is a local variable
        locals_ = node.locals
        for param in self.function.py_signature.parameters:
            if self.function.is_complex_param(param):
                value = param.type(self.function, param.name, args)
                # The params can inject IR at the beginning
                value.preamble(builder)
            else:
                value = self.function.preamble_for_param(builder, param, args)

            # Ok
            locals_[param.name] = value

        # Create the second block, this is where the code proper will start,
        # after allocation of the local variables.
        block_start = function.append_basic_block("start")
        builder.position_at_end(block_start)

        # Keep stuff we will need in this first pass
        self.function = function

        # Keep stuff for the second pass
        node.block_vars = block_vars
        node.block_start = block_start
        node.builder = builder
        node.f_rtype = ir_signature.return_type

    def Return_exit(self, node, parent, value):
        return True  # Means this statement terminates the block

    def If_test(self, node, parent, test, suffix_re=re.compile("if_true(.*)")):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        node.block_true = self.function.append_basic_block("if_true")
        node.block_suffix = suffix_re.match(node.block_true.name).group(1)

    def If_body(self, node, parent, body):
        node.block_false = self.function.append_basic_block("if_false" + node.block_suffix)

    def If_exit(self, node, parent, test, body, orelse):
        is_terminated = bool(body and body[-1] and orelse and orelse[-1])
        if not is_terminated:
            node.block_next = self.function.append_basic_block("if_next" + node.block_suffix)
        else:
            node.block_next = None

        return is_terminated

    def IfExp_test(self, node, parent, test):
        """
        IfExp(expr test, expr body, expr orelse)
        """
        node.block_true = self.function.append_basic_block("ifexp_true")

    def IfExp_body(self, node, parent, body):
        node.block_false = self.function.append_basic_block("ifexp_false")

    def IfExp_orelse(self, node, parent, orelse):
        node.block_next = self.function.append_basic_block("ifexp_next")

    def For_enter(self, node, parent):
        """
        For(expr target, expr iter, stmt* body, stmt* orelse)
        """
        assert not node.orelse, '"for ... else .." not supported'
        node.block_for = self.function.append_basic_block("for")
        node.block_body = self.function.append_basic_block("for_body")

    def For_exit(self, node, parent, *args):
        node.block_next = self.function.append_basic_block("for_out")

    def While_enter(self, node, parent):
        """
        While(expr test, stmt* body, stmt* orelse)
        """
        assert not node.orelse, '"while ... else .." not supported'
        node.block_while = self.function.append_basic_block("while")
        node.block_body = self.function.append_basic_block("while_body")

    def While_exit(self, node, parent, *args):
        node.block_next = self.function.append_basic_block("while_out")


class GenVisitor(NodeVisitor):
    """
    Builtin types are:
    identifier, int, string, bytes, object, singleton, constant

    singleton: None, True or False
    constant can be None, whereas None means "no value" for object.
    """

    function = None
    args = None
    builder = None
    f_rtype = None  # Type of the return value
    ltype = None  # Type of the local variable

    def print(self, line):
        print(self.depth * " " + line)

    def convert(self, value, type_):
        """
        Return the value converted to the given type.
        """
        return types.value_to_ir_value(self.builder, value, type_)

    #
    # Leaf nodes
    #

    def Name_visit(self, node, parent):
        """
        Name(identifier id, expr_context ctx)
        """
        name = node.id
        ctx = type(node.ctx)

        if ctx is ast.Load:
            return self.load(name)
        elif ctx is ast.Store:
            return name

        raise NotImplementedError(f"unexpected ctx={ctx}")

    def boolop_visit(self, node, parent):
        return type(node)

    def operator_visit(self, node, parent):
        return type(node)

    def unaryop_visit(self, node, parent):
        return type(node)

    def Eq_visit(self, node, parent):
        return "=="

    def NotEq_visit(self, node, parent):
        return "!="

    def Lt_visit(self, node, parent):
        return "<"

    def LtE_visit(self, node, parent):
        return "<="

    def Gt_visit(self, node, parent):
        return ">"

    def GtE_visit(self, node, parent):
        return ">="

    #
    # Literals
    #

    def List_exit(self, node, parent, elts, ctx):
        """
        List(expr* elts, expr_context ctx)
        """
        py_types = {type(x) for x in elts}
        n = len(py_types)
        if n == 0:
            # any type will do because the list is empty
            py_type = int
        elif n == 1:
            py_type = py_types.pop()
        else:
            raise TypeError("all list elements must be of the same type")

        el_type = types.type_to_ir_type(py_type)
        typ = ir.ArrayType(el_type, len(elts))
        return ir.Constant(typ, elts)

    #
    # Expressions
    #

    def FunctionDef_enter(self, node, parent):
        self.locals = node.locals
        self.builder = node.builder
        self.f_rtype = node.f_rtype
        self.block_vars = node.block_vars

    def FunctionDef_exit(self, node, parent, *args):
        if self.root.py_signature.return_type is None:
            if not self.builder.block.is_terminated:
                node.builder.ret_void()

        node.builder.position_at_end(node.block_vars)
        node.builder.branch(node.block_start)

    def BoolOp_exit(self, node, parent, op, values):
        """
        BoolOp(boolop op, expr* values)
        """
        ir_op = {
            ast.And: self.builder.and_,
            ast.Or: self.builder.or_,
        }[op]

        assert len(values) == 2
        left, right = values
        return ir_op(left, right)

    def BinOp_exit(self, node, parent, left, op, right):
        type_ = values_to_type(left, right)

        # Two Python values
        if not isinstance(type_, ir.Type):
            ast2op = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
            }
            py_op = ast2op.get(op)
            if py_op is None:
                raise NotImplementedError(
                    f"{op.__name__} operator for {type_} type not implemented"
                )
            return py_op(left, right)

        # Special case, power translated to math.pow
        left = self.convert(left, type_)
        right = self.convert(right, type_)
        if op == ast.Pow:
            return self.__call(math.pow, left, right)

        # One or more IR values
        d = {
            (ast.Add, ir.IntType): self.builder.add,
            (ast.Sub, ir.IntType): self.builder.sub,
            (ast.Mult, ir.IntType): self.builder.mul,
            (ast.Div, ir.IntType): self.builder.sdiv,
            (ast.Mod, ir.IntType): self.builder.srem,
            (ast.Add, ir.FloatType): self.builder.fadd,
            (ast.Sub, ir.FloatType): self.builder.fsub,
            (ast.Mult, ir.FloatType): self.builder.fmul,
            (ast.Div, ir.FloatType): self.builder.fdiv,
            (ast.Add, ir.DoubleType): self.builder.fadd,
            (ast.Sub, ir.DoubleType): self.builder.fsub,
            (ast.Mult, ir.DoubleType): self.builder.fmul,
            (ast.Div, ir.DoubleType): self.builder.fdiv,
        }
        base_type = type(type_)
        ir_op = d.get((op, base_type))
        if ir_op is None:
            raise NotImplementedError(f"{op.__name__} operator for {type_} type not implemented")

        return ir_op(left, right)

    def UnaryOp_exit(self, node, parent, op, operand):
        """
        UnaryOp(unaryop op, expr operand)
        """
        type_ = types.value_to_type(operand)
        if isinstance(type_, ir.Type):
            # IR value
            if op is ast.Not:
                op = self.builder.not_
            elif op is ast.USub:
                if isinstance(type_, ir.IntType):
                    op = self.builder.neg
                else:
                    op = self.builder.fneg
        else:
            # Python value
            op = {
                ast.Not: operator.not_,
                ast.USub: operator.neg,
            }[op]

        return op(operand)

    def IfExp_test(self, node, parent, test):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        self.builder.cbranch(test, node.block_true, node.block_false)
        self.builder.position_at_end(node.block_true)

    def IfExp_body(self, node, parent, body):
        if not self.builder.block.is_terminated:
            self.builder.branch(node.block_next)
        self.builder.position_at_end(node.block_false)

    def IfExp_orelse(self, node, parent, orelse):
        self.builder.branch(node.block_next)
        self.builder.position_at_end(node.block_next)

    def IfExp_exit(self, node, parent, test, body, orelse):
        """
        IfExp(expr test, expr body, expr orelse)
        """
        body = types.value_to_ir_value(self.builder, body)
        orelse = types.value_to_ir_value(self.builder, orelse)

        # The phi instruction has a type, so first we need to convert the
        # incoming edges to the same type.
        ltype = types.value_to_ir_type(body)
        rtype = types.value_to_ir_type(orelse)
        type_ = values_to_type(body, orelse)
        if ltype is not type_:
            with self.builder.goto_block(node.block_true):
                body = types.value_to_ir_value(self.builder, body, type_)
        if rtype is not type_:
            with self.builder.goto_block(node.block_false):
                orelse = types.value_to_ir_value(self.builder, orelse, type_)

        # The phi instruction
        phi = self.builder.phi(type_)
        phi.add_incoming(body, node.block_true)
        phi.add_incoming(orelse, node.block_false)
        return phi

    def Compare_exit(self, node, parent, left, ops, comparators):
        """
        Compare(expr left, cmpop* ops, expr* comparators)
        """
        assert len(ops) == 1
        assert len(comparators) == 1
        op = ops[0]
        right = comparators[0]

        type_ = values_to_type(left, right)

        # Two Python values
        if not isinstance(type_, ir.Type):
            ast2op = {
                "==": operator.eq,
                "!=": operator.ne,
                "<": operator.lt,
                "<=": operator.le,
                ">": operator.gt,
                ">=": operator.ge,
            }
            py_op = ast2op.get(op)
            return py_op(left, right)

        # At least 1 IR value
        left = self.convert(left, type_)
        right = self.convert(right, type_)

        d = {
            ir.IntType: self.builder.icmp_signed,
            ir.FloatType: self.builder.fcmp_unordered,  # XXX fcmp_ordered
            ir.DoubleType: self.builder.fcmp_unordered,  # XXX fcmp_ordered
        }
        base_type = type(type_)
        ir_op = d.get(base_type)
        return ir_op(op, left, right)

    def Index_exit(self, node, parent, value):
        """
        Index(expr value)
        """
        return value

    def Subscript_exit(self, node, parent, value, slice, ctx):
        """
        Subscript(expr value, slice slice, expr_context ctx)
        """
        # An smart object
        subscript = getattr(value, "subscript", None)
        if subscript is not None:
            return subscript(self, slice, ctx)

        # A pointer!
        if isinstance(value, ir.Value) and value.type.is_pointer:
            ptr = value
            ptr = self.builder.gep(ptr, [slice])
            return self.builder.load(ptr)

        raise NotImplementedError(f"{type(value)} does not support subscript []")

    def Tuple_exit(self, node, parent, elts, ctx):
        """
        Tuple(expr* elts, expr_context ctx)
        """
        assert ctx is ast.Load
        return elts

    #
    # if .. elif .. else
    #
    def If_test(self, node, parent, test):
        """
        If(expr test, stmt* body, stmt* orelse)
        """
        test = types.value_to_ir_value(self.builder, test, type_=types.int1)
        self.builder.cbranch(test, node.block_true, node.block_false)
        self.builder.position_at_end(node.block_true)

    def If_body(self, node, parent, body):
        if not self.builder.block.is_terminated:
            self.builder.branch(node.block_next)
        self.builder.position_at_end(node.block_false)

    def If_orelse(self, node, parent, orelse):
        if node.block_next is not None:
            if not self.builder.block.is_terminated:
                self.builder.branch(node.block_next)
            self.builder.position_at_end(node.block_next)

    #
    # for ...
    #
    def For_iter(self, node, parent, expr):
        """
        For(expr target, expr iter, stmt* body, stmt* orelse)
        """
        target = node.target.id
        if isinstance(expr, Range):
            start = expr.start
            stop = expr.stop
            node.step = expr.step
            name = target
        else:
            start = types.zero
            stop = ir.Constant(types.int64, expr.type.count)
            node.step = types.one
            name = "i"
            # Allocate and store the literal array to iterate
            arr = self.builder.alloca(expr.type)
            self.builder.store(expr, arr)

        # Allocate and initialize the index variable
        node.i = self.builder.alloca(stop.type, name=name)
        self.builder.store(start, node.i)  # i = start
        self.builder.branch(node.block_for)  # br %for

        # Stop condition
        self.builder.position_at_end(node.block_for)  # %for
        idx = self.builder.load(node.i)  # %idx = i
        test = self.builder.icmp_unsigned("<", idx, stop)  # %idx < stop
        self.builder.cbranch(test, node.block_body, node.block_next)  # br %test %body %next
        self.builder.position_at_end(node.block_body)  # %body

        # Keep variable to use within the loop
        if isinstance(expr, Range):
            self.locals[target] = idx
        else:
            ptr = self.builder.gep(arr, [types.zero, idx])  # expr[idx]
            x = self.builder.load(ptr)  # % = expr[i]
            self.locals[target] = x

    def For_exit(self, node, parent, *args):
        # Increment index variable
        a = self.builder.load(node.i)  # % = i
        b = self.builder.add(a, node.step)  # % = % + step
        self.builder.store(b, node.i)  # i = %
        # Continue
        self.builder.branch(node.block_for)  # br %for
        self.builder.position_at_end(node.block_next)  # %next

    #
    # while ...
    #
    def While_enter(self, node, parent):
        self.builder.branch(node.block_while)
        self.builder.position_at_end(node.block_while)

    def While_test(self, node, parent, test):
        self.builder.cbranch(test, node.block_body, node.block_next)
        self.builder.position_at_end(node.block_body)

    def While_exit(self, node, parent, *args):
        self.builder.branch(node.block_while)
        self.builder.position_at_end(node.block_next)

    #
    # Other non-leaf nodes
    #
    def Attribute_exit(self, node, parent, value, attr, ctx):
        """
        Attribute(expr value, identifier attr, expr_context ctx)
        """
        assert ctx is ast.Load
        value = getattr(value, attr)
        if isinstance(value, types.Node):
            value = value.Attribute_exit(self)

        if (
            isinstance(value, ir.Value)
            and value.type.is_pointer
            and not hasattr(value, "function_type")
        ):
            value = self.builder.load(value)

        return value

    def AnnAssign_annotation(self, node, parent, value):
        self.ltype = value

    def AnnAssign_exit(self, node, parent, target, annotation, value, simple):
        """
        AnnAssign(expr target, expr annotation, expr? value, int simple)
        """
        assert value is not None
        assert simple == 1

        ltype = types.type_to_ir_type(self.ltype)
        value = self.convert(value, ltype)
        self.ltype = None

        name = target
        try:
            ptr = self.lookup(name)
        except AttributeError:
            block_cur = self.builder.block
            self.builder.position_at_end(self.block_vars)
            ptr = self.builder.alloca(value.type, name=name)
            self.builder.position_at_end(block_cur)
            self.locals[name] = ptr

        return self.builder.store(value, ptr)

    def Assign_exit(self, node, parent, targets, value, *args):
        if len(targets) > 1:
            raise NotImplementedError("unpacking not supported")

        builder = self.builder
        value = types.value_to_ir_value(builder, value)

        target = targets[0]
        if type(target) is str:
            # x =
            name = target
            try:
                ptr = self.lookup(name)
            except AttributeError:
                block_cur = builder.block
                builder.position_at_end(self.block_vars)
                ptr = builder.alloca(value.type, name=name)
                builder.position_at_end(block_cur)
                self.locals[name] = ptr
        else:
            # x[i] =
            ptr = target
            # Convert type if needed
            ltype = target.type.pointee
            value = self.convert(value, ltype)

        return builder.store(value, ptr)

    def AugAssign_exit(self, node, parent, target, op, value):
        """
        AugAssign(expr target, operator op, expr value)
        """
        # Translate "a += b" to "a = a + b"
        left = self.load(target)
        value = self.BinOp_exit(node, parent, left, op, value)  # a + b
        return self.Assign_exit(node, parent, [target], value)  # a =

    def Return_enter(self, node, parent):
        self.ltype = self.f_rtype

    def Return_exit(self, node, parent, value):
        """
        Return(expr? value)
        """
        if value is None:
            assert self.f_rtype is types.void
            return self.builder.ret_void()

        value = self.convert(value, self.f_rtype)
        self.ltype = None
        return self.builder.ret(value)

    def __call(self, func, *args):
        type_ = types.value_to_ir_type(args[0])
        func = self.root.compiled.get((func, type_), func)

        # Mathematical functions expect floats, but the argument may be an
        # integer. This allows for instance to support math.cos(1)
        if not hasattr(func, "function_type"):
            type_ = types.float64
            func = self.root.compiled.get((func, type_), func)

        if not hasattr(func, "function_type"):
            raise TypeError(f"unexpected {func}")

        # Check the number of arguments is correct
        arg_types = func.function_type.args
        if len(args) != len(arg_types):
            n = len(arg_types)
            raise TypeError(
                f"{func.name} takes exactly one argument ({len(args)} given)"
                if n == 1
                else f"{func.name} expects {n} arguments, got {len(args)}"
            )

        # Convert to IR values of the correct type
        args = [
            types.value_to_ir_value(self.builder, arg, type_=arg_type)
            for arg, arg_type in zip(args, arg_types)
        ]

        return self.builder.call(func, args)

    def Call_exit(self, node, parent, func, args, keywords):
        """
        Call(expr func, expr* args, keyword* keywords)
        """
        assert not keywords
        if func is range:
            return Range(self.builder, *args)

        return self.__call(func, *args)


class Parameter:
    def __init__(self, name, type):
        self.name = name
        self.type = type


class Signature:
    def __init__(self, parameters, return_type):
        self.parameters = parameters
        self.return_type = return_type


class Function:
    """
    Wraps a Python function. Compiled to IR, it will be executed with libffi:

    f(...)

    Besides calling the function a number of attributes are available:

    name        -- the name of the function
    py_function -- the original Python function
    py_source   -- the source code of the Python function
    ir          -- LLVM's IR code
    """

    def __init__(self, llvm, py_function, signature, f_globals, optimize=True):
        assert type(py_function) is FunctionType
        self.llvm = llvm
        self.py_function = py_function
        self.name = py_function.__name__

        self.py_signature = self.get_py_signature(signature)
        self.compiled = False
        self.globals = f_globals
        self.optimize = optimize

    @staticmethod
    def is_complex_param(param):
        return type(param.type) is type and issubclass(param.type, types.ComplexType)

    def get_py_signature(self, signature):
        self.is_scalar = True
        inspect_signature = inspect.signature(self.py_function)
        if signature is not None:
            assert len(signature) == len(inspect_signature.parameters) + 1

        # Parameters
        params = []
        for i, name in enumerate(inspect_signature.parameters):
            param = inspect_signature.parameters[name]
            assert (
                param.kind <= inspect.Parameter.POSITIONAL_OR_KEYWORD
            ), "only positional arguments are supported"

            type_ = param.annotation if signature is None else signature[i]
            params.append(Parameter(name, type_))

        # The return type
        if signature is None:
            return_type = inspect_signature.return_annotation
        else:
            return_type = signature[-1]

        return Signature(params, return_type)

    def get_ir_signature(self, node, debug=0, *args):
        # (2) Infer return type if not given
        return_type = self.py_signature.return_type
        if return_type is inspect._empty:
            node.return_type = return_type
            InferVisitor(debug).traverse(node)
            return_type = node.return_type
            self.py_signature.return_type = return_type

        # (3) The IR signature
        nargs = len(args)
        params = []
        for i, param in enumerate(self.py_signature.parameters):
            name = param.name
            type_ = param.type
            # Get type from argument if not given explicitely
            arg = args[i] if i < nargs else None
            if type_ is inspect._empty:
                assert arg is not None
                type_ = types.value_to_ir_type(arg)
                self.py_signature.parameters[i] = Parameter(name, type_)

            # IR signature
            if type(type_) is type and issubclass(type_, types.ArrayType):
                dtype = type_.dtype
                dtype = types.type_to_ir_type(dtype).as_pointer()
                params.append(Parameter(name, dtype))
                for n in range(type_.ndim):
                    params.append(Parameter(f"{name}_{n}", types.int64))
            elif type(type_) is type and issubclass(type_, types.StructType):
                dtype = self.llvm.get_dtype(self.ir_module, type_)
                params.append(Parameter(name, dtype))
            elif getattr(type_, "__origin__", None) is typing.List:
                dtype = type_.__args__[0]
                dtype = types.type_to_ir_type(dtype).as_pointer()
                params.append(Parameter(name, dtype))
                params.append(Parameter(f"{name}_0", types.int64))
            else:
                dtype = types.type_to_ir_type(type_)
                params.append(Parameter(name, dtype))

        return_type = types.type_to_ir_type(return_type)
        ir_signature = Signature(params, return_type)
        return ir_signature

    def can_be_compiled(self):
        """
        Return whether we have the argument types, so we can compile the
        function.
        """
        for param in self.py_signature.parameters:
            if param.type is inspect._empty:
                return False

        return True

    def preamble(self, builder, args):
        pass

    def preamble_for_param(self, builder, param, args):
        return args[param.name]

    def compile(self, node=None, debug=0, *args):
        if node is None:
            self.py_source = inspect.getsource(self.py_function)
            # (1) Python AST
            if debug > 0:
                print("====== Source ======")
                print(self.py_source)

            node = ast.parse(self.py_source)
        else:
            self.py_source = None

        # (2) IR Signature
        self.ir_module = Module()
        ir_signature = self.get_ir_signature(node, debug, *args)
        self.ir_signature = ir_signature

        # (4) For libffi
        self.nargs = len(ir_signature.parameters)
        self.argtypes = [p.type for p in ir_signature.parameters]
        self.argtypes = [("p" if x.is_pointer else x.intrinsic_name) for x in self.argtypes]

        if ir_signature.return_type is types.void:
            self.rtype = ""
        elif ir_signature.return_type.is_pointer:
            self.rtype = "p"
        else:
            self.rtype = ir_signature.return_type.intrinsic_name

        # (5) Load functions
        node.compiled = {}
        signatures = {}
        for name in MATH_FUNCS:
            py_func = getattr(math, name)
            try:
                signature = inspect.signature(py_func)
            except ValueError:
                # inspect.signature(math.log) fails: a Python bug
                nargs = 1
            else:
                nargs = len(signature.parameters)

            for t in types.float32, types.float64:
                args = tuple(nargs * [t])
                signature = signatures.get(args)
                if signature is None:
                    signature = ir.FunctionType(t, args)
                    signatures[args] = signature
                fname = name if t is types.float64 else f"{name}f"
                func = ir.Function(self.ir_module, signature, name=fname)
                node.compiled[(py_func, args[0])] = func

        # (6) The IR module and function
        self.ir_function_type = ir.FunctionType(
            ir_signature.return_type, tuple(param.type for param in ir_signature.parameters)
        )
        ir_function = ir.Function(self.ir_module, self.ir_function_type, self.name)

        # (7) AST pass: structure
        node.globals = self.globals
        node.py_signature = self.py_signature
        node.ir_signature = ir_signature
        node.ir_function = ir_function

        if debug > 2:
            print("====== Debug: 1st pass ======")
        BlockVisitor(debug, self).traverse(node)

        # (8) AST pass: generate
        if debug > 2:
            print("====== Debug: 2nd pass ======")
        GenVisitor(debug, self).traverse(node)

        # (9) IR code
        if debug > 2:
            print("====== IR ======")
            print(self.ir)

        # Compile
        self.mod = self.llvm.compile_ir(self.ir, self.name, debug, self.optimize)

        # (11) Done
        self.compiled = True

    @property
    def ir(self):
        return str(self.ir_module)

    @property
    def bc(self):
        return self.mod.as_bitcode()

    def call_args(self, *args, debug=0):
        if self.compiled is False:
            self.compile(debug, *args)

        c_args = []
        for py_arg in args:
            c_type = self.ir_signature.parameters[len(c_args)].type
            for plugin in plugins:
                expand_argument = getattr(plugin, "expand_argument", None)
                if expand_argument is not None:
                    arguments = expand_argument(py_arg, c_type)
                    if arguments is not None:
                        c_args.extend(arguments)

        return tuple(c_args)

    def __call__(self, *args, debug=0):
        c_args = self.call_args(*args, debug=debug)

        value = self.call(c_args)
        if debug > 0:
            print("====== Output ======")
            print(f"args = {args}")
            print(f"ret  = {value}")

        return value

    def call(self, c_args):
        raise NotImplementedError


class LLVM:
    def __init__(self, fclass):
        self.fclass = fclass
        self.dtypes = {}

    def get_dtype(self, ir_module, type_):
        type_id = id(type_)
        if type_id in self.dtypes:
            return self.dtypes[type_id]

        dtype = ir_module.context.get_identified_type(type_._name_)
        dtype.set_body(*type_.get_body())
        dtype = dtype.as_pointer()
        self.dtypes[type_id] = dtype
        return dtype

    def jit(self, py_function=None, signature=None, ast=None, debug=0, optimize=True, lib=None):
        f_globals = {"math": math, "udf": udf}

        if type(py_function) is FunctionType:
            function = self.fclass(self, py_function, signature, f_globals, optimize)
            if function.can_be_compiled():
                function.compile(node=ast, debug=debug)
            return function

        # Called as a decorator
        def wrapper(py_function):
            function = self.fclass(self, py_function, signature, f_globals, optimize)
            if function.can_be_compiled():
                function.compile(node=ast, debug=debug)
            if function.is_scalar:
                # Register scalar UDFs only
                ia.udf_registry[lib] = function
            return function

        return wrapper

    def compile_ir(self, llvm_ir, name, debug, optimize=True):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """

        # Create a LLVM module object from the IR
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        # Assign triple, so the IR can be saved and compiled with llc
        if debug > 1:
            print("====== IR (parsed) ======")
            print(mod)

        # Optimize
        if optimize:
            pmb = binding.PassManagerBuilder()
            pmb.opt_level = 2  # 0-3 (default=2)
            pmb.loop_vectorize = True

            mpm = binding.ModulePassManager()
            # Needed for automatic vectorization
            triple = binding.get_process_triple()
            target = binding.Target.from_triple(triple)
            tm = target.create_target_machine()
            tm.add_analysis_passes(mpm)

            pmb.populate(mpm)
            mpm.run(mod)

            if debug > 1:
                print("====== IR (optimized) ======")
                print(mod)

        return mod

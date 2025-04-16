import ast 
from ast import * 
import copy
import pdb
import ast

def find_all_frees(tree):
    class FindFrees(ast.NodeVisitor):
        def __init__(self):
            self.free_vars = set()

        def get_free_vars(self, func_node):
            class FreeFinder(ast.NodeVisitor):
                def __init__(self): 
                    self.used_vars = set()
                    self.bound_vars = set()

                def visit_arg(self, node):
                    self.bound_vars.add(node.arg)

                def visit_Name(self, node): 
                    if node.id not in ("print", "id", "eval", "input", "int"):
                        if isinstance(node.ctx, ast.Load): 
                            self.used_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Store): 
                            self.bound_vars.add(node.id)
                
                def get_free_vars(self):
                    return self.used_vars - self.bound_vars  

            if isinstance(func_node, (ast.FunctionDef, ast.Lambda)): 
                finder = FreeFinder()
                finder.visit(func_node)
                return finder.get_free_vars()

            return set()   

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            self.free_vars.update(self.get_free_vars(node))

        def visit_Lambda(self, node):
            self.generic_visit(node) 
            self.free_vars.update(self.get_free_vars(node))

    freefinder = FindFrees()
    freefinder.visit(tree)
    return freefinder.free_vars
 
def heapify(ast_tree, free_vars):
    class Heapifier(ast.NodeTransformer):
        def __init__(self, free_vars):
            self.free_vars = set(free_vars)  

        def get_free_vars(self, func_node):
            class FreeFinder(ast.NodeVisitor):
                def __init__(self): 
                    self.used_vars = set()
                    self.bound_vars = set()

                def visit_arg(self, node):
                    self.bound_vars.add(node.arg)

                def visit_Name(self, node): 
                    if node.id not in ("print", "id", "eval", "input", "int"):
                        if isinstance(node.ctx, ast.Load): 
                            self.used_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Store): 
                            self.bound_vars.add(node.id)

                def visit_FunctionDef(self, node):
                    self.bound_vars.add(node.name)
                    
                    
                
                def get_free_vars(self):
                    return self.used_vars - self.bound_vars  
                

            if isinstance(func_node, (ast.FunctionDef, ast.Lambda)): 
                finder = FreeFinder()
                finder.visit(func_node)
                return finder.get_free_vars()

            return set()   
        
        def visit_Name(self, node):
            if node.id in self.free_vars:
                if isinstance(node.ctx, ast.Load):  
                    return ast.Subscript(
                        value=ast.Name(id=node.id, ctx=ast.Load()),  
                        slice=ast.Constant(value=0),  
                        ctx=ast.Load()
                    )
            return node  

        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.free_vars:
                node.targets[0] = ast.Subscript(
                        value=node.targets[0],  
                        slice=ast.Constant(value=0),  
                        ctx=ast.Load()
                    )
            elif isinstance(node.targets[0], ast.Name) and node.targets[0].id not in self.free_vars: 
                self.generic_visit(node)
            return node 
        
        def visit_FunctionDef(self, node):
            ext_free_vars = self.free_vars
            self.free_vars = self.get_free_vars(node)
            self.generic_visit(node)
            self.free_vars = ext_free_vars
            return node
        def visit_Lambda(self, node):
            ext_free_vars = self.free_vars
            self.free_vars = self.get_free_vars(node)
            self.generic_visit(node)
            self.free_vars = ext_free_vars
            return node
        
    def pre_heapify(ast_tree, free_vars): 
        pre_heaps = []
        for free_var in free_vars:
            pre_heap = ast.Assign(
                    targets = [ast.Name(id = free_var, ctx = ast.Store())], 
                    value=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load())
                )
            pre_heaps.append(pre_heap)
        ast_tree.body = pre_heaps + ast_tree.body
        return ast_tree
            
    heapified = Heapifier(free_vars).visit(ast_tree)
    return pre_heapify(heapified, free_vars)

def flatpy_closure(ast_tree):
    add_funcs_src = """
def create_closure(func, frees):
    return func, frees

def get_fun_ptr(closure):
    return closure[0]

def get_free_vars(closure):
    return closure[1]
        """
    add_funcs_ast = ast.parse(add_funcs_src)

    ret_ast = copy.deepcopy(ast_tree)

    class IfBodyFixer(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)  # Recursively visit nested nodes
            if not node.body:
                node.body = [ast.Pass()]
            if not node.orelse:
                node.orelse = []  # Optional: ensure orelse is explicitly a list
            return node

    ret_ast = IfBodyFixer().visit(ret_ast)
    ast.fix_missing_locations(ret_ast)

    ret_ast.body = add_funcs_ast.body + ret_ast.body
    return ret_ast

def closure_conversion(ast_tree, global_frees):

    class Closure_Conversion(ast.NodeTransformer): 
        def __init__(self, global_frees):
            self.function_count = 0 
            self.transformed_funcs = [] 
            self.func_map = {}
            self.global_frees = global_frees

        def get_free_vars(self, func_node):
            class FreeFinder(ast.NodeVisitor):
                def __init__(self): 
                    self.used_vars = set()
                    self.bound_vars = set()
                
                def ignore(self, string):
                    if string in ["print", "id", "eval", "input", "int", "create_closure", "get_fun_ptr", "get_free_vars", "set_subscript"] or string.startswith("Lambda_"):
                        return True
                    return False
                    

                def visit_arg(self, node):
                    self.bound_vars.add(node.arg)

                def visit_Name(self, node):
                    if not self.ignore(node.id):
                        if isinstance(node.ctx, ast.Load): 
                            self.used_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Store): 
                            self.bound_vars.add(node.id)

                def get_free_vars(self):
                    return self.used_vars - self.bound_vars  

            if isinstance(func_node, (ast.FunctionDef, ast.Lambda)): 
                finder = FreeFinder()
                finder.visit(func_node)
                return finder.get_free_vars()

            return set() 
        
        def closure_prod(self, lambda_name, free_vars, func_name):
            _elts = []
            for free in free_vars: 
                # if free == func_name or free in self.global_frees: 
                #     elt = ast.Subscript(
                #         value=ast.Name(id=free, ctx=ast.Load()),  
                #         slice=ast.Constant(value=0),  
                #         ctx=ast.Load()
                #     )
                # else:
                elt = ast.Name(id=free, ctx=ast.Load())
                _elts.append(elt)

            _args = [
                ast.Name(id=lambda_name, ctx=ast.Load()), 
                ast.List(elts=_elts, ctx=ast.Load())  
            ]
            
            _name = ast.Name(id="create_closure", ctx=ast.Load())

            return ast.Call(func=_name, args=_args, keywords=[])

        def lambda_prod(self, pass_args, _body, free_vars): 
            func_name = f"Lambda_{self.function_count}"
            _args = ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=f"free_vars{self.function_count}", annotation=None)] + pass_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            )

            ret_func = ast.FunctionDef(
                name=func_name,
                args=_args,
                body=_body,
                decorator_list=[]
            )

            assignments = []

            
            for i, free_var in enumerate(free_vars): 
                _assign = ast.Assign(
                    targets=[ast.Name(id=free_var, ctx=ast.Store())], 
                    value=ast.Subscript(
                        value=ast.Name(id=f"free_vars{self.function_count}", ctx=ast.Load()),  
                        slice=ast.Constant(value=i),  
                        ctx=ast.Load()
                    )
                )
                assignments.append(_assign)

            # for free_var in free_vars: 
            #     pre_heap = ast.Assign(
            #         targets = [ast.Name(id = free_var, ctx = ast.Store())], 
            #         value=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load())
            #     )
            #     self.transformed_funcs.append(pre_heap)

            

            self.function_count += 1  
            ret_func.body = assignments + ret_func.body  

            self.transformed_funcs.append(ret_func) 

            return func_name

        def func_call_prod(self, func_name, args):
            get_fun_call = ast.Call(
                func=ast.Name(id="get_fun_ptr", ctx=Load()),
                args=[func_name],
                keywords=[]
            )
            get_frees_call = ast.Call(
                func=ast.Name(id="get_free_vars", ctx=Load()),
                args=[func_name],
                keywords=[]
            )
            call_prod = ast.Call(
                func=get_fun_call,
                args=[get_frees_call] + args,
                keywords=[]
            )
            return call_prod
        
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            free_vars = self.get_free_vars(node)
            func_name = self.lambda_prod(node.args.args, node.body, free_vars)
            self.func_map[node.name] = func_name
            closure_call = self.closure_prod(func_name, free_vars, node.name)
            if node.name in free_vars or node.name in self.global_frees: 
                target = ast.Subscript(
                        value=ast.Name(id=node.name, ctx=ast.Load()),  
                        slice=ast.Constant(value=0),  
                        ctx=ast.Store()
                    )
            else: 
                target = ast.Name(id=node.name, ctx=ast.Store())

            ret_assign = ast.Assign(
                targets=[target],
                value=closure_call
            )


            return ret_assign
        
        def visit_Lambda(self, node):
            self.generic_visit(node)
            free_vars = self.get_free_vars(node)
            func_name = self.lambda_prod(node.args.args, [ast.Return(value=node.body)], free_vars)
            return  self.closure_prod(func_name, free_vars, "")

        def visit_Call(self, node): 
            self.generic_visit(node)
            if isinstance(node.func, Name) and node.func.id in ["print", "id", "eval", "input", "int", "set_subscript"]: 
                return node
            else: 
                new_call = self.func_call_prod(node.func, node.args)
                return new_call

    transformer = Closure_Conversion(global_frees)
    transformed_tree = transformer.visit(ast_tree)

    if isinstance(transformed_tree, ast.Module):
        transformed_tree.body = transformer.transformed_funcs +  transformed_tree.body

    return ast.fix_missing_locations(transformed_tree)

def closure_flattener(ast_tree):
    class ClosureFlattener(ast.NodeTransformer):
        def __init__(self):
            self.tmp_counter = 0

        def new_tmp_name(self):
            res = f'c_temp_{self.tmp_counter}'
            self.tmp_counter += 1
            return res
        

        def undocall(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, Name) and node.value.func.id == "create_closure":
                if isinstance(node.value.args[1], ast.List):
                    temp_name = self.new_tmp_name()
                    temp_var = ast.Name(id=temp_name, ctx=ast.Store())
                    temp_load = ast.Name(id=temp_name, ctx=ast.Load())
                    assign_temp = ast.Assign(targets=[temp_var], value=node.value.args[1])
                    node.value.args[1] = temp_load
                    return [assign_temp, node]
            return [node]
        
        def visit_Assign(self, node): 
            temped = self.undocall(node)
            return temped

    return ClosureFlattener().visit(ast_tree)

def uniquify_frees(ast_tree):
    class UniquifyFrees(ast.NodeTransformer):

        def __init__(self): 
            super().__init__()
            self.func_ct = 0

        def get_free_vars(self, func_node):
            class FreeFinder(ast.NodeVisitor):
                def __init__(self): 
                    self.used_vars = set()
                    self.bound_vars = set()

                def visit_arg(self, node):
                    self.bound_vars.add(node.arg)

                def visit_Name(self, node): 
                    if node.id not in ("print", "id", "eval", "input", "int"):
                        if isinstance(node.ctx, ast.Load): 
                            self.used_vars.add(node.id)
                        elif isinstance(node.ctx, ast.Store): 
                            self.bound_vars.add(node.id)
                
                def get_free_vars(self):
                    return self.used_vars - self.bound_vars  

            if isinstance(func_node, (ast.FunctionDef, ast.Lambda)):
                finder = FreeFinder()
                finder.visit(func_node)
                return finder.get_free_vars()

            return set() 
        
        def rename_bound_vars(self, node, free_vars):
            class BoundVarRenamer(ast.NodeTransformer):
                def __init__(self, free_vars, func_ct):
                    super().__init__()
                    self.free_vars = free_vars
                    self.func_ct = func_ct

                def visit_Name(self, node):
                    if node.id not in ("print", "id", "eval", "input", "int"):
                        if node.id not in self.free_vars:
                            if not node.id.endswith(f"_b{self.func_ct}"):
                                return ast.copy_location(ast.Name(id=node.id + f"_b{self.func_ct}", ctx=node.ctx), node)
                    return node
                
                def visit_arg(self, node): 
                    if node.arg not in self.free_vars:
                        node.arg = node.arg + f"_b{self.func_ct}"
                    return node

            return BoundVarRenamer(free_vars, self.func_ct).visit(node)

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            free_vars = self.get_free_vars(node)
            print(f"Free variables in function: {free_vars}")
            node = self.rename_bound_vars(node, free_vars)
            self.func_ct += 1
            return node
        
        def visit_Lambda(self, node):
            self.generic_visit(node) 
            free_vars = self.get_free_vars(node)
            print(f"Free variables in lambda: {free_vars}")
            node = self.rename_bound_vars(node, free_vars)
            self.func_ct += 1
            return node

    return UniquifyFrees().visit(ast_tree)

def unique_valid_PO(tree):
    class Uniquify(ast.NodeTransformer):
        def __init__(self): 
            self.ignore = ["print", "id", "eval", "input", "int"]
        
        def visit_Name(self, node):
            if (node.id not in self.ignore):
                node.id = f"s_{node.id}"
            return node
        
        def visit_arg(self, node):
            node.arg = f"s_{node.arg}"
            return node
        
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.name  = f"s_{node.name}"
            return node

    transformer = Uniquify()
    unique = transformer.visit(tree)
    return unique

def flatten(tree: ast.Module):

    """Converts an `ast.AST` to an AST of simple expressions."""
    tmp_counter = 0
    suite_stack = [[]]

    def new_tmp_name():
        nonlocal tmp_counter
        res = f'tmp_{tmp_counter}'
        tmp_counter += 1
        return res

    def simple(n):
        return isinstance(n, (Name, Constant))

    def get_simple(n):
        if simple(n):
            return n
        result = rec(n)
        return result if simple(result) else rec_with_tmp(result)

    def _append(n):
        suite_stack[-1].append(n)

    def rec_with_tmp(n):
        sub = rec(n)
        tmp_name = new_tmp_name()
        _append(Assign(targets=[Name(id=tmp_name, ctx=Store())], value=sub, type_comment=None))
        return Name(id=tmp_name, ctx=Load())

    def rec(n):
        match n:
            case Name() | Constant():
                return n
            case Assign(targets=[Subscript(value=val, slice=slice_node)], value=value):
                if not simple(val):
                    val = rec_with_tmp(val)
                value = get_simple(value)
                slice_node = get_simple(slice_node)
                call_node = Call(func=Name(id='set_subscript', ctx=Load()), args=[val, slice_node, value], keywords=[])
                _append(Expr(value=call_node))
            case Assign(value=value):
                n.value = rec(value)
                _append(n)
            case Expr(value=value):
                if not isinstance(value, (Call)):
                    pass
                else:
                    n.value = rec(value)
                    _append(n)
            case UnaryOp(operand=operand):
                n.operand = rec_with_tmp(operand) if not simple(operand) else operand
            case BinOp(left=left, right=right):
                n.left = rec_with_tmp(left) if not simple(left) else left
                n.right = rec_with_tmp(right) if not simple(right) else right
            case Call(func=func, args=args):
                if not simple(func):
                    n.func = get_simple(func)
                    func = n.func
                if  not (isinstance(func, Name) and func.id in ("eval", "int")):
                    for i, arg in enumerate(args): 
                        if not simple(arg):
                            args[i] = get_simple(arg)
                if func.id in {'eval', 'int'}:
                    args[0] = rec(args[0])
            case If(test=test, body=body, orelse=orelse):
                n.test = get_simple(test)
                suite_stack.append([])
                for stmt in body: rec(stmt)
                n.body = suite_stack.pop()
                suite_stack.append([])
                for stmt in orelse: rec(stmt)
                n.orelse = suite_stack.pop()
                _append(n)
            case Compare(left=left, comparators=comparators):
                n.left = get_simple(left)
                n.comparators[0] = get_simple(comparators[0])
            case While(test=test, body=body, orelse=orelse):
                suite_stack.append([])
                rec(If(test=test, body=[], orelse=[Break()]))
                n.test = Constant(value=1)
                for stmt in body: rec(stmt)
                n.body = suite_stack.pop()
                suite_stack.append([])
                for stmt in orelse: rec(stmt)
                n.orelse = suite_stack.pop()
                _append(n)
            case BoolOp(values=values):
                n.values = [rec(v) for v in values]
            case Break():
                _append(n)
            case Subscript(value=value, slice=slice_node):
                n.value = get_simple(value)
                n.slice = get_simple(slice_node)
            case List(elts=elts):
                n.elts = [get_simple(e) for e in elts]
            case Dict(keys=keys, values=values):
                n.keys = [get_simple(k) for k in keys]
                n.values = [get_simple(v) for v in values]
            case FunctionDef(name = name, args = args, body = body):
                suite_stack.append([])
                for func_body in body: rec(func_body)
                n.body = suite_stack.pop()
                _append(n)
            case Return(value = value):
                n.value = get_simple(value)
                _append(n)

        return n

    for stmt in tree.body:
        rec(stmt)
    tree.body = [ast.fix_missing_locations(node) for node in suite_stack[0]]
    return tree

def flat_lists(flat_tree):
    class UnNestList(ast.NodeTransformer):
        def __init__(self):
            self.tmp_counter = 0
        def new_tmp_name(self):
            res = f'listtmp_{self.tmp_counter}'
            self.tmp_counter += 1
            return res
        def list_unnester(self, node):
            new_nodes = []
            for i, elt in enumerate(node.value.elts):
                if isinstance(elt, List):
                    tmp_name = Name(id=self.new_tmp_name(), ctx=Store())
                    new_nodes.append(Assign(targets=[tmp_name], value=elt))
                    node.value.elts[i] = tmp_name
            return new_nodes + [node]
        def visit_Assign(self, node):
            if isinstance(node.value, List):
                return self.list_unnester(node)
            return node
    class FlatList(ast.NodeTransformer):
        def list_flattener(self, node):
            if isinstance(node.value, List):
                list_size = len(node.value.elts)
                create_list_node = Call(func=Name(id='create_list', ctx=Load()), args=[Constant(value=list_size)], keywords=[])
                new_nodes = [Assign(targets=node.targets, value=create_list_node)]
                for i, elt in enumerate(node.value.elts):
                    set_call = Call(func=Name(id='set_subscript', ctx=Load()), args=[node.targets[0], Constant(value=i), elt], keywords=[])
                    new_nodes.append(Expr(value=set_call))
                return new_nodes

        def visit_Assign(self, node):
            if isinstance(node.value, List):
                return self.list_flattener(node)
            return node
    transformer = UnNestList()
    flat_tree = transformer.visit(flat_tree)
    ast.fix_missing_locations(flat_tree)
    transformer = FlatList()
    flat_list = transformer.visit(flat_tree)
    ast.fix_missing_locations(flat_list)
    return flat_list

def flat_dicts(flat_tree): 
    class FlatDict(ast.NodeTransformer):
        def dict_flattener(self, node):
            if isinstance(node.value, Dict):
                dict_size = len(node.value.keys)
                create_dict_node = Call(func=Name(id='create_dict', ctx=Load()), args=[Constant(value=dict_size)], keywords=[])
                new_nodes = [Assign(targets=node.targets, value=create_dict_node)]
                for i, key in enumerate(node.value.keys):
                    set_call = Call(func=Name(id='dict_subscript', ctx=Load()), args=[key, node.value.values[i], node.targets[0]], keywords=[])
                    new_nodes.append(Expr(value=set_call))
                return new_nodes
        def visit_Assign(self, node):
            if isinstance(node.value, Dict):
                return self.dict_flattener(node)
            return node
    transformer = FlatDict()
    flat_dict = transformer.visit(flat_tree)
    ast.fix_missing_locations(flat_dict)
    return flat_dict

def subscript_remover(flat_tree):
    class SubscriptRemover(ast.NodeTransformer):
        def __init__(self):
            self.tmp_counter = 0

        def new_tmp_name(self):
            res = f'subtmp_{self.tmp_counter}'
            self.tmp_counter += 1
            return res

        def visit_Assign(self, node):
            if isinstance(node.targets[0], ast.Subscript):
                temp_name = self.new_tmp_name()
                temp_var = ast.Name(id=temp_name, ctx=ast.Store())
                temp_load = ast.Name(id=temp_name, ctx=ast.Load())

                # Create a temporary variable to store the value
                assign_temp = ast.Assign(targets=[temp_var], value=node.value)

                # Replace the original subscript with a set_subscript call
                set_subscript_call = ast.Expr(
                    value=ast.Call(
                    func=ast.Name(id='set_subscript', ctx=ast.Load()),
                    args=[node.targets[0].value, node.targets[0].slice, temp_load],
                    keywords=[]
                    )
                )

                return [assign_temp, set_subscript_call]

            return node
        
    transformer = SubscriptRemover()
    flat_tree = transformer.visit(flat_tree)
    ast.fix_missing_locations(flat_tree)
    return flat_tree

def explicate(flat_ast):
    tmp_count =0 
    def ltemp():
        """generate new temporary variable name"""
        nonlocal tmp_count
        res = f'ltmp_{tmp_count}'
        tmp_count += 1
        return res   
    
    def _append(n): suite_stack[len(suite_stack) -1].append(n)
    
    def construct_is_int(n): return ast.Call(func=ast.Name(id='is_int', ctx=ast.Load()), args=[n], keywords=[])
    
    def construct_is_bool(n): return ast.Call(func=ast.Name(id='is_bool', ctx=ast.Load()), args=[n], keywords=[])
    
    def construct_is_big(n): return ast.Call(func=ast.Name(id='is_big', ctx=ast.Load()), args=[n], keywords=[])
    
    def construct_if(check, _body, _else): return ast.If(test=check, body=_body, orelse=_else)

    def prod_proj(n, var_type): 
        match var_type:
            case "int": return ast.Call(func=ast.Name(id='project_int', ctx=ast.Load()), args=[n], keywords=[])
            case "bool": return ast.Call(func=ast.Name(id='project_bool', ctx=ast.Load()), args=[n], keywords=[])
            case "big" | "list" | "dict": return ast.Call(func=ast.Name(id='project_big', ctx=ast.Load()), args=[n], keywords=[])
    
    def prod_inj(n, var_type):
        match var_type:
            case "int": return ast.Call(func=ast.Name(id='inject_int', ctx=ast.Load()), args=[n], keywords=[])
            case "bool": return ast.Call(func=ast.Name(id='inject_bool', ctx=ast.Load()), args=[n], keywords=[])
            case "big" | "list" | "dict": return ast.Call(func=ast.Name(id='inject_big', ctx=ast.Load()), args=[n], keywords=[])
    
    def inject_const(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                n = ast.Constant(value=int(n.value))
                return prod_inj(n, "bool")
            elif isinstance(n.value, int):
                return prod_inj(n, "int")
        
    def bin_op_unbox(left, right, assign, op): 

        #For now additions between bools and integers are the exact same, same with proj, what we are checking for are two lists. 
        # Let
        left_tmp = ltemp()
        right_tmp = ltemp()
        if isinstance(left, ast.Constant):
            left = inject_const(left)
        if isinstance(right, ast.Constant):
            right = inject_const(right)

        _append(ast.Assign(targets=[ast.Name(id=left_tmp, ctx=ast.Store())], value=left))
        _append(ast.Assign(targets=[ast.Name(id=right_tmp, ctx=ast.Store())], value=right))
        l_tmp = ast.Name(id=left_tmp, ctx=ast.Load())
        r_tmp = ast.Name(id=right_tmp, ctx=ast.Load())

        #Explication Body
        # FOR INTEGERS

        l_tmp1 = ast.Name(id=ltemp(), ctx=ast.Load())
        r_tmp1 = ast.Name(id=ltemp(), ctx=ast.Load())
        val_tmp = ast.Name(id=ltemp(), ctx=ast.Load())

        left_proj_int = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "int"))
        right_proj_int = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "int"))

        left_proj_bool = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "bool"))
        right_proj_bool = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "bool"))

        left_proj_big = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "big"))
        right_proj_big = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "big"))


        inj_body = ast.BinOp(left=l_tmp1, op=op, right=r_tmp1)
        big_inj_body = ast.Call(func=ast.Name(id = 'add', ctx=ast.Load()), args=[l_tmp1, r_tmp1], keywords=[])
        big_inject = prod_inj(val_tmp, "big")
        inject = prod_inj(val_tmp, "int")

        

        int_body =[]
        int_body.append(ast.Assign(targets=[val_tmp], value=inj_body))
        int_body.append(ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject))

        big_body = []
        big_body.append(ast.Assign(targets=[val_tmp], value=big_inj_body))
        big_body.append(ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=big_inject))

        int_bool_body = [left_proj_int] + [right_proj_bool] + int_body
        bool_int_body = [left_proj_bool] + [right_proj_int] + int_body
        bool_bool_body = [left_proj_bool] + [right_proj_bool] + int_body
        int_int_body = [left_proj_int] + [right_proj_int] + int_body
        big_big_body = [left_proj_big] + [right_proj_big] + big_body

        left_proj_bool = prod_proj(l_tmp, "bool")
        right_proj_bool = prod_proj(r_tmp, "bool")

        # Conditional construction (explication tree)
        # If includes check for bool OR int. 
        # if_right_bool = construct_if(construct_is_bool(r_tmp), int_bool_body, [])
        # if_right = construct_if(construct_is_int(r_tmp), int_int_body, [if_right_bool])
        # if_left_bool_right_int = construct_if(construct_is_int(l_tmp), int_int_body, [])
        # if_right_bool_2 = construct_if(construct_is_bool(r_tmp), bool_bool_body, [if_left_bool_right_int])
        # if_left_bool = construct_if(construct_is_bool(l_tmp), [if_right_bool_2], [])
        # if_left = construct_if(construct_is_int(l_tmp), [if_right], [if_left_bool])\

        # trying again.
        value_error = ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=ast.Name(id = "#ValueError", ctx=ast.Load()))
        if_left_big_right_big = construct_if(construct_is_big(r_tmp), big_big_body, [value_error])
        if_left_big = construct_if(construct_is_big(l_tmp), [if_left_big_right_big], [value_error])
        if_left_bool_right_bool = construct_if(construct_is_bool(r_tmp), bool_bool_body, [value_error])
        if_left_bool_right_int = construct_if(construct_is_int(r_tmp), bool_int_body, [if_left_bool_right_bool])
        if_left_bool = construct_if(construct_is_bool(l_tmp), [if_left_bool_right_int], [if_left_big])
        if_left_int_right_bool = construct_if(construct_is_bool(r_tmp), int_bool_body, [value_error])
        if_left_int_right_int = construct_if(construct_is_int(r_tmp), int_int_body, [if_left_int_right_bool])
        if_left_int = construct_if(construct_is_int(l_tmp), [if_left_int_right_int], [if_left_bool])
        _append(if_left_int)

    def compare_unbox(left, right, assign, op, ret_type):
        left_tmp = ltemp()
        right_tmp = ltemp()
        if isinstance(left, ast.Constant):
            left = inject_const(left)
        if isinstance(right, ast.Constant):
            right = inject_const(right)

        _append(ast.Assign(targets=[ast.Name(id=left_tmp, ctx=ast.Store())], value=left))
        _append(ast.Assign(targets=[ast.Name(id=right_tmp, ctx=ast.Store())], value=right))
        l_tmp = ast.Name(id=left_tmp, ctx=ast.Load())
        r_tmp = ast.Name(id=right_tmp, ctx=ast.Load())

        if isinstance(op, ast.Is):
            compare_tmp = ast.Name(id=ltemp(), ctx=ast.Load())
            compare_body = ast.Assign(targets=[compare_tmp], value=ast.Compare(left=l_tmp, ops=[Eq()], comparators=[r_tmp]))
            inject = prod_inj(compare_tmp, ret_type)
            _append(compare_body)
            _append(ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject))
            return

        l_tmp1 = ast.Name(id=ltemp(), ctx=ast.Load())
        r_tmp1 = ast.Name(id=ltemp(), ctx=ast.Load())
        val_tmp = ast.Name(id=ltemp(), ctx=ast.Load())
        compare_tmp = ast.Name(id=ltemp(), ctx=ast.Load())

        left_proj_int = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "int"))
        right_proj_int = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "int"))

        left_proj_bool = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "bool"))
        right_proj_bool = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "bool"))

        left_proj_big = ast.Assign(targets=[l_tmp1], value=prod_proj(l_tmp, "big"))
        right_proj_big = ast.Assign(targets=[r_tmp1], value=prod_proj(r_tmp, "big"))

        compare_body = ast.Assign(targets=[compare_tmp], value=ast.Compare(left=l_tmp1, ops=[op], comparators=[r_tmp1]))
        inject = prod_inj(compare_tmp, ret_type)

        int_body = [
            left_proj_int,
            right_proj_int,
            compare_body,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject)
        ]

        int_bool_body = [
            left_proj_int,
            right_proj_bool,
            compare_body,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject)
        ]

        bool_int_body = [
            left_proj_bool,
            right_proj_int,
            compare_body,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject)
        ]

        bool_body = [
            left_proj_bool,
            right_proj_bool,
            compare_body,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject)
        ]

        big_body = [
            left_proj_big,
            right_proj_big,
            ast.Assign(
            targets=[compare_tmp],
            value=ast.Call(
                func=ast.Name(id="equal" if isinstance(op, ast.Eq) else "not_equal", ctx=ast.Load()),
                args=[l_tmp1, r_tmp1],
                keywords=[]
            )
            ),
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=prod_inj(compare_tmp, ret_type))
        ]

        value_error = ast.Assign(
            targets=[ast.Name(id=assign, ctx=ast.Store())],
            value=ast.Name(id="#ValueError", ctx=ast.Load())
        )

        if_left_big_right_big = construct_if(construct_is_big(r_tmp), big_body, [value_error])
        if_left_big = construct_if(construct_is_big(l_tmp), [if_left_big_right_big], [value_error])
        if_left_bool_right_bool = construct_if(construct_is_bool(r_tmp), bool_body, [value_error])
        if_left_bool_right_int = construct_if(construct_is_int(r_tmp), bool_int_body, [if_left_bool_right_bool])
        if_left_bool = construct_if(construct_is_bool(l_tmp), [if_left_bool_right_int], [if_left_big])
        if_left_int_right_bool = construct_if(construct_is_bool(r_tmp), int_bool_body, [value_error])
        if_left_int_right_int = construct_if(construct_is_int(r_tmp), int_body, [if_left_int_right_bool])
        if_left_int = construct_if(construct_is_int(l_tmp), [if_left_int_right_int], [if_left_bool])
        _append(if_left_int)

    def if_unbox(value, assign):
        val_tmp = ltemp()
        if isinstance(value, ast.Constant):
            value = inject_const(value)
        _append(ast.Assign(targets=[ast.Name(id=val_tmp, ctx=ast.Store())], value=value))

        # projections
        proj_int = prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "int")
        proj_bool = prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "bool")
        proj_big = prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "big")

        # Temporary variables for big projections
        proj_big_temp = ltemp()
        empty_list_temp = ltemp()
        empty_dict_temp = ltemp()
        not_equal_list_temp = ltemp()
        not_equal_dict_temp = ltemp()

        # Create empty list and dict
        create_empty_list = ast.Assign(
            targets=[ast.Name(id=empty_list_temp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='create_list', ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[]
            )
        )
        create_empty_dict = ast.Assign(
            targets=[ast.Name(id=empty_dict_temp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='create_dict', ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[]
            )
        )

        # Assign projection for big
        assign_proj_big = ast.Assign(
            targets=[ast.Name(id=proj_big_temp, ctx=ast.Store())],
            value=proj_big
        )

        # Compare with empty list and dict
        not_equal_list = ast.Assign(
            targets=[ast.Name(id=not_equal_list_temp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="not_equal", ctx=ast.Load()),
                args=[
                    ast.Name(id=proj_big_temp, ctx=ast.Load()),
                    ast.Name(id=empty_list_temp, ctx=ast.Load())
                ],
                keywords=[]
            )
        )
        not_equal_dict = ast.Assign(
            targets=[ast.Name(id=not_equal_dict_temp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="not_equal", ctx=ast.Load()),
                args=[
                    ast.Name(id=proj_big_temp, ctx=ast.Load()),
                    ast.Name(id=empty_dict_temp, ctx=ast.Load())
                ],
                keywords=[]
            )
        )

        # Final assignment for big
        final_assign_big = ast.If(
            test=ast.Name(id=not_equal_list_temp, ctx=ast.Load()),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=assign, ctx=ast.Store())],
                    value=ast.Name(id=not_equal_dict_temp, ctx=ast.Load())
                )
            ],
            orelse=[
                ast.Assign(
                    targets=[ast.Name(id=assign, ctx=ast.Store())],
                    value=ast.Name(id=not_equal_list_temp, ctx=ast.Load())
                )
            ]
        )

        # Construct if statements
        if_big = construct_if(
            construct_is_big(ast.Name(id=val_tmp, ctx=ast.Load())),
            [
                assign_proj_big,
                create_empty_list,
                create_empty_dict,
                not_equal_list,
                not_equal_dict,
                final_assign_big
            ],
            [ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=ast.Name(id="#ValueError", ctx=ast.Load()))]
        )
        if_bool = construct_if(
            construct_is_bool(ast.Name(id=val_tmp, ctx=ast.Load())),
            [ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=proj_bool)],
            [if_big]
        )
        if_int = construct_if(
            construct_is_int(ast.Name(id=val_tmp, ctx=ast.Load())),
            [ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=proj_int)],
            [if_bool]
        )
        _append(if_int)

    def unop_unbox(value, assign, op):
        val_tmp = ltemp()

        if isinstance(value, ast.Constant):
            value = inject_const(value)

        _append(ast.Assign(targets=[ast.Name(id=val_tmp, ctx=ast.Store())], value=value))

        store_tmp = ast.Name(id=ltemp(), ctx=ast.Load())

        calc_tmp = ltemp()
        val_proj_int = ast.Name(id=calc_tmp, ctx=ast.Load())
        val_proj_bool = ast.Name(id=calc_tmp, ctx=ast.Load())
        val_proj_big = ast.Name(id=calc_tmp, ctx=ast.Load())

        # projections
        proj_int = ast.Assign(targets=[store_tmp], value=prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "int"))
        proj_bool = ast.Assign(targets=[store_tmp], value=prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "bool"))
        proj_big = ast.Assign(targets=[store_tmp], value=prod_proj(ast.Name(id=val_tmp, ctx=ast.Load()), "big"))

        do_calc = ast.Assign(targets=[ast.Name(id=calc_tmp, ctx=Load())], value=ast.UnaryOp(op=op, operand=store_tmp))

        # unary operation
        inj_int = ast.Name(id=calc_tmp, ctx=ast.Load())

        # Injection
        inject_int = prod_inj(inj_int, "int")

        int_body = [
            proj_int,
            do_calc,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject_int)
        ]
        bool_body = [
            proj_bool,
            do_calc,
            ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=inject_int)
        ]

        value_error = ast.Assign(
            targets=[ast.Name(id=assign, ctx=ast.Store())],
            value=ast.Name(id="#ValueError", ctx=ast.Load())
        )
        if_bool = construct_if(construct_is_bool(ast.Name(id=val_tmp, ctx=ast.Load())), bool_body, [value_error])
        if_int = construct_if(construct_is_int(ast.Name(id=val_tmp, ctx=ast.Load())), int_body, [if_bool])

        # Append the final conditional
        _append(if_int)
  
    def call_unbox(n):   
        if isinstance(n.value, ast.Call):
            if isinstance(n.value.func, ast.Name):
                if n.value.func.id in ('create_list', 'create_dict'):
                    if isinstance(n.value.args[0], ast.Constant):
                        tmp = ltemp()
                        arg_type = type(n.value.args[0].value).__name__
                        ass_to = prod_inj(n.value.args[0], arg_type)
                        _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                        n.value.args[0] = ast.Name(id=tmp, ctx=ast.Load())
                    store_tmp = ltemp()

                    _append(ast.Assign(targets=[ast.Name(id=store_tmp, ctx=ast.Store())], value=n.value))
                    injected_call = prod_inj(ast.Name(id = store_tmp, ctx = Load()), "big")
                    _append(ast.Assign(targets = n.targets, value = injected_call))
                    n.value = ast.Name(id=store_tmp, ctx=ast.Load())
                elif n.value.func.id == "get_subscript":
                    for i, arg in enumerate(n.value.args):
                        if isinstance(arg, ast.Constant):
                            tmp = ltemp()
                            arg_type = type(arg.value).__name__
                            ass_to = prod_inj(arg, arg_type)
                            _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                            n.value.args[i] = ast.Name(id=tmp, ctx=ast.Load())
                    _append(n)
                elif n.value.func.id == "int":
                    if isinstance(n.value.args[0], Compare):
                        tmp = ltemp()
                        compare_unbox(n.value.args[0].left, n.value.args[0].comparators[0], n.targets[0].id, n.value.args[0].ops[0], 'int')
                        n.value.args[0] = ast.Name(id=tmp, ctx=ast.Load())
                    elif isinstance(n.value.args[0], UnaryOp):
                        not_unbox(n.value.args[0].operand, n.targets[0].id, 'int')
                    elif isinstance(n.value.args[0], ast.Constant):
                        tmp = ltemp()
                        arg_type = type(n.value.args[0].value).__name__
                        ass_to = prod_inj(n.value.args[0], arg_type)
                        _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                        n.value.args[0] = ast.Name(id=tmp, ctx=ast.Load())

                elif n.value.func.id == "create_closure":
                    for i, arg in enumerate(n.value.args):
                        if isinstance(arg, ast.Constant):
                            tmp = ltemp()
                            arg_type = type(arg.value).__name__
                            ass_to = prod_inj(arg, arg_type)
                            _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                            n.value.args[i] = ast.Name(id=tmp, ctx=ast.Load())
                    store_tmp = ltemp()
                    _append(ast.Assign(targets=[ast.Name(id=store_tmp, ctx=ast.Store())], value=n.value))
                    injected_call = prod_inj(ast.Name(id=store_tmp, ctx=ast.Load()), "big")
                    _append(ast.Assign(targets=n.targets, value=injected_call))

                else:
                    for i, arg in enumerate(n.value.args):
                        if isinstance(arg, ast.Constant):
                            tmp = ltemp()
                            arg_type = type(arg.value).__name__
                            ass_to = prod_inj(arg, arg_type)
                            _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                            n.value.args[i] = ast.Name(id=tmp, ctx=ast.Load())
                    _append(n)
    
    def not_unbox(operand, assign, ret_type):
        operand_tmp = ltemp()

        if isinstance(operand, ast.Constant):
            operand = inject_const(operand)

        _append(ast.Assign(targets=[ast.Name(id=operand_tmp, ctx=ast.Store())], value=operand))
        operand_name = ast.Name(id=operand_tmp, ctx=ast.Load())

        # Projections
        proj_int = prod_proj(operand_name, "int")
        proj_bool = prod_proj(operand_name, "bool")
        proj_big = prod_proj(operand_name, "big")

        proj_tmp = ltemp()

        # Assignments
        ass_int = ast.Assign(targets=[ast.Name(id=proj_tmp, ctx=ast.Store())], value=proj_int)
        ass_bool = ast.Assign(targets=[ast.Name(id=proj_tmp, ctx=ast.Store())], value=proj_bool)
        ass_big = ast.Assign(targets=[ast.Name(id=proj_tmp, ctx=ast.Store())], value=proj_big)

        compare_tmp = ltemp()

        # Comparisons
        compare_int = ast.Assign(
            targets=[ast.Name(id=compare_tmp, ctx=ast.Store())],
            value=ast.Compare(left=ast.Name(id=proj_tmp, ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=0)])
        )
        compare_bool = ast.Assign(
            targets=[ast.Name(id=compare_tmp, ctx=ast.Store())],
            value=ast.Compare(left=ast.Name(id=proj_tmp, ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=0)])
        )

        # For big,
        empty_list_tmp = ltemp()
        create_empty_list = ast.Assign(
            targets=[ast.Name(id=empty_list_tmp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='create_list', ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[]
            )
        )
        empty_dict_tmp = ltemp()
        create_empty_dict = ast.Assign(
            targets=[ast.Name(id=empty_dict_tmp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='create_dict', ctx=ast.Load()),
                args=[ast.Constant(value=0)],
                keywords=[]
            )
        )
        not_equal_list_tmp = ltemp()
        not_equal_list = ast.Assign(
            targets=[ast.Name(id=not_equal_list_tmp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="not_equal", ctx=ast.Load()),
                args=[
                    ast.Name(id=proj_tmp, ctx=ast.Load()),
                    ast.Name(id=empty_list_tmp, ctx=ast.Load())
                ],
                keywords=[]
            )
        )
        not_equal_dict_tmp = ltemp()
        not_equal_dict = ast.Assign(
            targets=[ast.Name(id=not_equal_dict_tmp, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="not_equal", ctx=ast.Load()),
                args=[
                    ast.Name(id=proj_tmp, ctx=ast.Load()),
                    ast.Name(id=empty_dict_tmp, ctx=ast.Load())
                ],
                keywords=[]
            )
        )
        final_assign = ast.If(
            test=ast.Name(id=not_equal_list_tmp, ctx=ast.Load()),
            body=[ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=prod_inj(ast.Name(id=not_equal_list_tmp, ctx=ast.Load()), ret_type))],
            orelse=[ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=prod_inj(ast.Name(id=not_equal_dict_tmp, ctx=ast.Load()), ret_type))]
        )

        # Conditional bodies
        int_body = [ass_int, compare_int, ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=prod_inj(ast.Name(id=compare_tmp, ctx=ast.Load()), ret_type))]
        bool_body = [ass_bool, compare_bool, ast.Assign(targets=[ast.Name(id=assign, ctx=ast.Store())], value=prod_inj(ast.Name(id=compare_tmp, ctx=ast.Load()), ret_type))]
        big_body = [
            ass_big,
            create_empty_list,
            create_empty_dict,
            not_equal_list,
            not_equal_dict,
            final_assign
        ]

        # Value error assignment
        value_error = ast.Assign(
            targets=[ast.Name(id=assign, ctx=ast.Store())],
            value=ast.Name(id="#ValueError", ctx=ast.Load())
        )

        # Conditional construction
        if_big = construct_if(construct_is_big(operand_name), big_body, [value_error])
        if_bool = construct_if(construct_is_bool(operand_name), bool_body, [if_big])
        if_int = construct_if(construct_is_int(operand_name), int_body, [if_bool])

        # Append the final conditional
        _append(if_int)

    def check_subscript(n):
        if isinstance(n, ast.Subscript):
            if isinstance(n.value, ast.Constant):
                tmp_value = ltemp()
                value_type = type(n.value.value).__name__
                injected_value = prod_inj(n.value, value_type)
                _append(ast.Assign(targets=[ast.Name(id=tmp_value, ctx=ast.Store())], value=injected_value))
                n.value = ast.Name(id=tmp_value, ctx=ast.Load())

            if isinstance(n.slice, ast.Constant):
                tmp_slice = ltemp()
                slice_type = type(n.slice.value).__name__
                injected_slice = prod_inj(n.slice, slice_type)
                _append(ast.Assign(targets=[ast.Name(id=tmp_slice, ctx=ast.Store())], value=injected_slice))
                n.slice = ast.Name(id=tmp_slice, ctx=ast.Load())

            return ast.Call(
                func=Name(id='get_subscript', ctx=Load()),
                args=[n.value, n.slice],
                keywords=[]
            )
        else:
            return n
    def rec(n): 
        if isinstance(n, ast.Assign):
            if isinstance(n.value, ast.Call):
                call_unbox(n)
            elif isinstance(n.value, ast.Subscript):
                n.value = ast.Call(
                    func=Name(id='get_subscript', ctx=Load()),
                    args=[n.value.value, n.value.slice],
                    keywords=[]
                )
                call_unbox(n)
            elif isinstance(n.value, Name):
                _append(n)
            elif isinstance(n.value, ast.BinOp):
                n.value.left = check_subscript(n.value.left)
                n.value.right = check_subscript(n.value.right)
                bin_op_unbox(n.value.left, n.value.right, n.targets[0].id, n.value.op)
            elif isinstance(n.value, ast.Compare):
                compare_unbox(n.value.left, n.value.comparators[0], n.targets[0].id, n.value.ops[0], 'bool')
            elif isinstance(n.value, ast.UnaryOp):
                if isinstance(n.value.op, ast.Not):
                    not_unbox(n.value.operand, n.targets[0].id, 'bool')
                else:
                    unop_unbox(n.value.operand, n.targets[0].id, n.value.op)
            elif isinstance(n.value, ast.Constant):
                if isinstance(n.value.value, bool):
                    n.value.value = int(n.value.value)
                    n.value = prod_inj(n.value, "bool")
                    # print(ast.unparse(n.value))
                    _append(n)
                elif isinstance(n.value.value, int):
                    n.value = prod_inj(n.value, "int")
                    _append(n)
            else: 
                _append(n)

        elif isinstance(n, ast.If):
            test_temp = ltemp()
            if_unbox(n.test, test_temp)
            n.test = ast.Name(id=test_temp, ctx=ast.Load())
            if_body = []
            suite_stack.append(if_body)
            for do in n.body:
                rec(do)
            n.body = suite_stack.pop()
            else_body = []
            suite_stack.append(else_body)
            for doelse in n.orelse: 
                rec(doelse)
            n.orelse = suite_stack.pop()
            _append(n)

        elif isinstance(n, FunctionDef):
            func_body = []
            suite_stack.append(func_body)
            for do in n.body:
                rec(do)
            n.body = suite_stack.pop()
            _append(n)
        elif isinstance(n, While):
            while_body = []
            suite_stack.append(while_body)
            for do in n.body:
                rec(do)
            n.body = suite_stack.pop()
            else_body = []
            suite_stack.append(else_body)
            for doelse in n.orelse: 
                rec(doelse)
            n.orelse = suite_stack.pop()
            _append(n)
        elif isinstance(n, ast.Expr):
            if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name):
                if n.value.func.id == 'print':
                    if isinstance(n.value.args[0], ast.Constant):
                        arg_type = type(n.value.args[0].value).__name__
                        tmp = ltemp()
                        ass_to = prod_inj(n.value.args[0], arg_type)
                        _append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=ass_to))
                        n.value.args[0] = ast.Name(id=tmp, ctx=ast.Load())

                if n.value.func.id in ('set_subscript', 'dict_subscript'):
                    if isinstance(n.value.args[0], ast.Constant):
                        tmp0 = ltemp()
                        ass_to = inject_const(n.value.args[0])
                        _append(ast.Assign(targets=[ast.Name(id=tmp0, ctx=ast.Store())], value=ass_to))
                        n.value.args[0] = ast.Name(id=tmp0, ctx=ast.Load())
                    if isinstance(n.value.args[1], ast.Constant):
                        tmp1 = ltemp()
                        ass_to = inject_const(n.value.args[1])
                        _append(ast.Assign(targets=[ast.Name(id=tmp1, ctx=ast.Store())], value=ass_to))
                        n.value.args[1] = ast.Name(id=tmp1, ctx=ast.Load())
                    if isinstance(n.value.args[2], ast.Constant):
                        tmp2 = ltemp()
                        ass_to = inject_const(n.value.args[2])
                        _append(ast.Assign(targets=[ast.Name(id=tmp2, ctx=ast.Store())], value=ass_to))
                        n.value.args[2] = ast.Name(id=tmp2, ctx=ast.Load())
                    
                    # Reorder arguments for dict_subscript
                    if n.value.func.id == 'dict_subscript':
                        n.value.args = [n.value.args[2], n.value.args[0], n.value.args[1]]
                    
            _append(n)
        elif isinstance(n, ast.Return):
            if isinstance(n.value, ast.Constant):
                temp_name = ltemp()
                arg_type = type(n.value.value).__name__
                injected_value = inject_const(n.value)
                _append(ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=injected_value))
                _append(ast.Return(value=ast.Name(id=temp_name, ctx=ast.Load())))
            else:
                _append(n)
        else:
            _append(n)
 
    main_suite = []
    suite_stack = [main_suite]
    soln = []
    for line in flat_ast.body:
        rec(line)
    flat_ast.body = [ast.fix_missing_locations(node) for node in suite_stack[0]]
    return flat_ast

def runtime(explicated): 
    runtimes = ast.parse("""
def is_int(x):
    return isinstance(x, int)
def is_bool(x):
    return isinstance(x, bool)
def is_big(x):
    isinstance(x, int)
def project_int(x):
    return x
def project_bool(x):
    return x
def project_big(x):
    return x
def inject_int(x):
    return x
def inject_bool(x):
    return x
def inject_big(x):
    return x  
def equal_pyobj(x,y):
    return x == y
def create_list(x): 
    return []
       
              """)
    runtimes.body.extend(explicated.body)
    return runtimes

def cond_nest(tree):
    class NestAnd(ast.NodeTransformer):
        def nest(self, node, ind1, ind2):
            return ast.BoolOp(
                op=node.op,
                values=[node.values[ind1], node.values[ind2]]
            )
        def visit_BoolOp(self, node):
            if isinstance(node.op, ast.And):
                while(len(node.values) > 2):
                    nest = self.nest(node, len(node.values) - 2, len(node.values) - 1)
                    node.values.pop()
                    node.values.pop()
                    node.values.append(nest)
                cond_nest(node.values[0])
                cond_nest(node.values[1])
                return node
            if isinstance(node.op, ast.Or):
                while(len(node.values) > 2):    
                    nest = self.nest(node, len(node.values) - 2, len(node.values) - 1)
                    node.values.pop()
                    node.values.pop()
                    node.values.append(nest)
                cond_nest(node.values[0])
                cond_nest(node.values[1])
                return node
    transformer = NestAnd()
    nested = transformer.visit(tree)
    ast.fix_missing_locations(nested)
    return nested

def desugar(tree):
    class RidBinary(ast.NodeTransformer):
        def __init__(self):
            self.tmp_counter = 0
            self.bool_op_count = 0

        def new_temp(self):
            temp_name = f"sweet{self.tmp_counter}"
            self.tmp_counter += 1
            return temp_name

        def visit_Assign(self, node):
            return self._handle_expr_stmt(node, is_return=False)

        def visit_Return(self, node):
            return self._handle_expr_stmt(node, is_return=True)

        def _handle_expr_stmt(self, node, is_return):
            expr = node.value
            if not isinstance(expr, (ast.BoolOp, ast.IfExp)):
                return node  # No desugaring needed

            self.bool_op_count += 1
            temp_name = self.new_temp()
            temp_var = ast.Name(id=temp_name, ctx=ast.Load())
            store_temp = ast.Name(id=temp_name, ctx=ast.Store())

            # Determine the condition expression
            if isinstance(expr, ast.BoolOp):
                test_expr = expr.values[0]
                alt_expr = expr.values[1]
                if isinstance(expr.op, ast.Or):
                    body_val = test_expr
                    else_val = alt_expr
                else:  
                    body_val = alt_expr
                    else_val = test_expr
            elif isinstance(expr, ast.IfExp):
                test_expr = expr.test
                body_val = expr.body
                else_val = expr.orelse

            stmts = []

            # If the test is complex, evaluate it once and assign to a temp
            if not isinstance(test_expr, (ast.Name, ast.Constant)):
                test_temp = self.new_temp()
                test_var = ast.Name(id=test_temp, ctx=ast.Load())
                test_store = ast.Name(id=test_temp, ctx=ast.Store())
                stmts.append(ast.Assign(targets=[test_store], value=test_expr))
                test_expr = test_var  # Replace test with the temp variable

            if isinstance(expr, ast.BoolOp):
                if isinstance(expr.op, ast.Or):
                    body_val = test_expr
                    else_val = alt_expr
                else:  
                    body_val = alt_expr
                    else_val = test_expr
            elif isinstance(expr, ast.IfExp):
                test_expr = expr.test
                body_val = expr.body
                else_val = expr.orelse

            if_stmt = ast.If(
                test=test_expr,
                body=[ast.Assign(targets=[store_temp], value=body_val)],
                orelse=[ast.Assign(targets=[store_temp], value=else_val)]
            )
            stmts.append(if_stmt)

            # Add final return or assign
            if is_return:
                stmts.append(ast.Return(value=temp_var))
            else:
                stmts.append(ast.Assign(targets=node.targets, value=temp_var))

            return stmts

    transformer = RidBinary()
    desugared = transformer.visit(tree)
    ast.fix_missing_locations(desugared)
    return transformer.bool_op_count

def flat_calls(tree):

    class FlatCalls(ast.NodeTransformer):
        def __init__(self):
            self.tmp_counter = 0

        def new_tmp_name(self, prefix):
            res = f'{prefix}_{self.tmp_counter}'
            self.tmp_counter += 1
            return res

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Call):
                if isinstance(node.value.func.func, ast.Name) and node.value.func.func.id == 'get_fun_ptr':
                    temp_fun_ptr = self.new_tmp_name("callptr")
                    temp_free_vars = self.new_tmp_name("callfree")

                    # Assign temp for get_fun_ptr
                    assign_fun_ptr = ast.Assign(
                        targets=[ast.Name(id=temp_fun_ptr, ctx=ast.Store())],
                        value=node.value.func
                    )

                    # Assign temp for get_free_vars
                    assign_free_vars = ast.Assign(
                        targets=[ast.Name(id=temp_free_vars, ctx=ast.Store())],
                        value=node.value.args[0]
                    )

                    # Replace the original call with the un-nested version
                    new_call = ast.Call(
                        func=ast.Name(id=temp_fun_ptr, ctx=ast.Load()),
                        args=[
                            ast.Name(id=temp_free_vars, ctx=ast.Load()),
                            *node.value.args[1:]
                        ],
                        keywords=[]
                    )

                    return [assign_fun_ptr, assign_free_vars, ast.Expr(value=new_call)]

            return self.generic_visit(node)

        def visit_Assign(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Call):
                if isinstance(node.value.func.func, ast.Name) and node.value.func.func.id == 'get_fun_ptr':
                    temp_fun_ptr = self.new_tmp_name("callptr")
                    temp_free_vars = self.new_tmp_name("callfree")

                    # Assign temp for get_fun_ptr
                    assign_fun_ptr = ast.Assign(
                        targets=[ast.Name(id=temp_fun_ptr, ctx=ast.Store())],
                        value=node.value.func
                    )

                    # Assign temp for get_free_vars
                    assign_free_vars = ast.Assign(
                        targets=[ast.Name(id=temp_free_vars, ctx=ast.Store())],
                        value=node.value.args[0]
                    )

                    # Replace the original call with the un-nested version
                    new_call = ast.Call(
                        func=ast.Name(id=temp_fun_ptr, ctx=ast.Load()),
                        args=[
                            ast.Name(id=temp_free_vars, ctx=ast.Load()),
                            *node.value.args[1:]
                        ],
                        keywords=[]
                    )

                    return [assign_fun_ptr, assign_free_vars, ast.Assign(targets=node.targets, value=new_call)]

            return self.generic_visit(node)

    transformer = FlatCalls()
    flat_tree = transformer.visit(tree)
    ast.fix_missing_locations(flat_tree)
    return flat_tree
def func_flattener(tree):
    class FlattenFunctionReturns(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    flattened_return = flatten(ast.Module(body=[stmt], type_ignores=[]))
                    new_body.extend(flattened_return.body)
                else:
                    new_body.append(stmt)
            node.body = new_body
            return node

    transformer = FlattenFunctionReturns()
    flattened_tree = transformer.visit(tree)
    ast.fix_missing_locations(flattened_tree)
    return flattened_tree
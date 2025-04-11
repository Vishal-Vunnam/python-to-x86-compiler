#!/usr/bin/env python3.10
import ast
from ast import *
import pdb
class x86:

    def __init__(self):
        self.body = []  
    def add_instruction(self, instr, loc1, loc2):
        self.body.append({"instr": instr, "loc1": loc1, "loc2": loc2})
    def add_at(self, instr, loc1, loc2, index):
        if index < 0 or index >= len(self.body):
            raise ValueError("Instruction index out of range")
        self.body.insert(index + 1, {"instr": instr, "loc1": loc1, "loc2": loc2})
    def __repr__(self):
        return "\n".join(
            f"{item['instr']}, {item['loc1']}, {item['loc2']}" 
            for item in self.body)
    def __len__(self):
        return len(self.body)
    def remove_instruction(self, index):
        if index < 0 or index >= len(self.body):
            raise ValueError("Instruction index out of range")
        del self.body[index]
    def replace_instruction(self, index, instr, loc1, loc2):
        if index < 0 or index >= len(self.body):
            raise ValueError("Instruction index out of range")
        self.body[index] = {"instr": instr, "loc1": loc1, "loc2": loc2}
    def get_instr(self): 
        return item['instr']
    
class Ctrl_Graph: 
    def __init__(self):
        self.vertices = {}  
        self.edges = {}     # Each key is a vertex, and value is a list of connected vertices
    
    def add_vertex(self, vertex, instr_set):
        self.vertices[vertex] = instr_set
        if vertex not in self.edges:
            self.edges[vertex] = []
    
    def add_edge(self, vertex, edge): 
        if vertex in self.edges:
            if edge not in self.edges[vertex]:
                self.edges[vertex].append(edge)
        else:
            self.edges[vertex] = [edge]
    
    def __repr__(self):
        output = "Control Graph:\n"
        for vertex, instr_set in self.vertices.items():
            output += f"Vertex: {vertex}\n"
            output += f"{instr_set}\n"
            output += f"Edges: {self.edges.get(vertex, [])}\n\n"
        return output
    
    def print_graph(self):
        """Prints the entire control graph, including all vertices with their x86 instruction sets and connected edges."""
        print(self.__repr__())
    
    def print_vertex(self, vertex):
        """Prints the details of a specific vertex, including its instruction set and connected edges."""
        if vertex in self.vertices:
            print(f"Vertex: {vertex}")
            print("Instruction Set:")
            print(self.vertices[vertex])
            print(f"Edges: {self.edges.get(vertex, [])}")
        else:
            print("Vertex not found.")        
    
class InterferenceGraph:
    def __init__(self):
        self.vertices = {}  
        self.edges = {}
        self.vertices["eax"] = "apple"
        self.vertices["ecx"] = "carrot"
        self.vertices["edx"] = "donut"
        self.edges["eax"] = []
        self.edges["ecx"] = []
        self.edges["edx"] = []
        
    def add_vertex(self, vertex, color="blank"):
        self.vertices[vertex] = color
        if vertex not in self.edges:
            self.edges[vertex] = [] 

    def change_color(self, vertex, color):
        if vertex in self.vertices:
            self.vertices[vertex] = color
        # else:
        #     raise KeyError(f"Vertex {vertex} not found in the graph.")

    def add_edge(self, u, v):
        if v == u: 
            return 
        if u not in self.vertices:
            return 
        if v not in self.vertices:
            return 
        if v not in self.edges[u]:
            self.edges[u].append(v)
        if u not in self.edges[v]:
            self.edges[v].append(u)

    def in_vertex(self, vertex): 
        return vertex in self.vertices

    def __repr__(self):
        result = "Vertices:\n"
        for vertex, color in self.vertices.items():
            result += f"  {vertex}: {color}\n"
        result += "Edges:\n"
        for vertex, neighbors in self.edges.items():
            result += f"  {vertex} -> {', '.join(neighbors)}\n"
        return result
    
    def highest_order(self):
        max_edges = -1
        vertex_with_max_edges = None
        for vertex, neighbors in self.edges.items():
            if len(neighbors) > max_edges:
                max_edges = len(neighbors)
                vertex_with_max_edges = vertex
        return vertex_with_max_edges
    def get_edges(self, vertex):
        if vertex in self.edges:
            return self.edges[vertex]
        else:
            raise KeyError(f"Vertex {vertex} not found in the graph.")
    def get_vertices(self): 
        return self.vertices
            
    def get_color(self, vertex):
        return self.vertices.get(vertex, "no color??")
        
    
    def get_first_blank(self):
        for vertex, color in self.vertices.items():
            if color == "blank":
                return vertex
        return None  # Return None if no vertex has "blank" as its color
    def get_spills(self):
        return [vertex for vertex in self.vertices if vertex.startswith("sc_temp")]
    
def simple_expr_to_x86_ir(tree): 
    def simple(n):
        return isinstance(n, Name | Constant)
    ir = x86()
    cont_count = 0
    curr_while = []
    curr_irs = []
    curr_irs.append(ir)
    func_irs = []
    
    def add_instruction(instr, loc1, loc2):
        nonlocal curr_irs
        curr_irs[-1].add_instruction(instr, loc1, loc2)
        
        
    def ir_rec(n):
        nonlocal cont_count, curr_irs, func_irs
        if isinstance(n, Module):
            for child in n.body:
                ir_rec(child)
                
        elif isinstance(n, Assign):
            if simple(n.value):
                target_loc = ir_rec(n.targets[0])
                value_loc = ir_rec(n.value)
                add_instruction("movl", value_loc, target_loc)
                
            if isinstance(n.value, BinOp):
                target_loc = ir_rec(n.targets[0])
                left = ir_rec(n.value.left)
                right = ir_rec(n.value.right)
                op = ir_rec(n.value.op)
                if(target_loc == left):
                    add_instruction("addl", right, target_loc)
                elif (target_loc == right):
                    add_instruction("addl", left, target_loc)
                else:
                    add_instruction("movl", left, target_loc)
                    add_instruction("addl", right, target_loc)
                
            if isinstance(n.value, UnaryOp): 
                target_loc = ir_rec(n.targets[0])
                operand = ir_rec(n.value.operand)
                op = ir_rec(n.value.op)
                add_instruction("movl", operand, target_loc)
                add_instruction(op, target_loc, "")
                
            if isinstance(n.value, Compare): 
                target_loc = ir_rec(n.targets[0])
                left = ir_rec(n.value.left)
                comparator = ""
                if n.value.comparators[0]:
                    comparator = ir_rec(n.value.comparators[0])
                if isinstance(n.value.ops[0], Eq):
                    add_instruction("equals", left, comparator )
                if isinstance(n.value.ops[0], NotEq):
                    add_instruction("nequals", left, comparator )
                add_instruction("movl", "#bool", target_loc)

            if isinstance(n.value, Call):
                if n.value.func.id in ("eval", "int"):
                    target_loc = ir_rec(n.targets[0])
                    val_loc = ir_rec(n.value)
                    add_instruction("movl", val_loc, target_loc)

                elif n.value.func.id in ("create_list", "create_dict", "equal", "not_equal", "get_subscript", "add", "is_true" ):
                    target_loc = ir_rec(n.targets[0])
                    val_loc = ir_rec(n.value)
                    add_instruction("movl", val_loc, target_loc)
                
                elif n.value.func.id.startswith("inject_"):
                    target_loc = ir_rec(n.targets[0])
                    val_loc = ir_rec(n.value)
                    add_instruction("movl", val_loc, target_loc)
                elif n.value.func.id.startswith("project_"):
                    target_loc = ir_rec(n.targets[0])
                    val_loc = ir_rec(n.value)
                    add_instruction("movl", val_loc, target_loc)

                else:
                    target_loc = ir_rec(n.targets[0])
                    val_loc = ir_rec(n.value)
                    add_instruction("movl", val_loc, target_loc)

        elif isinstance(n, Expr):
            if (isinstance(n.value, Call)):
                arg_string = ", ".join(ir_rec(arg) for arg in n.value.args)
                add_instruction("call", arg_string, n.value.func.id)
        elif isinstance(n, Name):
            return f"{n.id}"
        elif isinstance(n, BinOp):
            op = ir_rec(n.op)
            left = ir_rec(n.left)
            right = ir_rec(n.right)
            add_instruction(op, left, right)
        elif isinstance(n, UnaryOp):
            op = ir_rec(n.op)
            operand = ir_rec(n.operand)
            add_instruction(op, ir_rec(n.operand), "") 
            return operand
        elif isinstance(n, Compare): 
            left = ir_rec(n.left)
            comparator = ""
            if n.comparators[0]:
                comparator = ir_rec(n.comparators[0])
            if isinstance(n.ops[0], Eq):
                add_instruction("equals", left, comparator ) 
            if isinstance(n.ops[0], NotEq):
                add_instruction("nequals", left, comparator ) 
        elif isinstance(n, Call):
            arg_string = ", ".join(ir_rec(arg) for arg in n.args if isinstance(arg, (Name, Constant)))

            if n.func.id == "is_true":
                add_instruction("call", ir_rec(n.args[0]), "is_true")
                return "#bool"
            
            elif n.func.id in {"create_list", "create_dict"}:
                add_instruction("call", ir_rec(n.args[0]), n.func.id)
                return f"#{n.func.id.split('_')[1]}"
            
            elif n.func.id in {"equal", "not_equal"}:
                add_instruction("call", arg_string, n.func.id)
                return "#bool"
            
            elif n.func.id == "get_subscript":
                add_instruction("call", arg_string, "get_subscript")
                return "#subscript"
            
            elif n.func.id == "add":
                add_instruction("call", arg_string, "add")
                return "#list"
            
            elif n.func.id == "eval":
                add_instruction("call", "eval_input", "")
                return "#input"
                
            elif n.func.id == "int":
                ir_rec(n.args[0])
                return "#int"
            
            elif n.func.id.startswith("is_"):
                add_instruction("call", ir_rec(n.args[0]), n.func.id)
                return "#type_chk"
                
            elif n.func.id.startswith("inject_"):
                add_instruction("call", ir_rec(n.args[0]), n.func.id)
                return "#val_inj"
                
            elif n.func.id.startswith("project_"):
                add_instruction("call", ir_rec(n.args[0]), n.func.id)
                return "#val_proj"
            
            else:
                add_instruction("call", arg_string, n.func.id)
                return f"#ret_{n.func.id}" 
        elif isinstance(n, If): 
            control_count = cont_count
            cont_count += 1
            add_instruction("cmpl", "$0", ir_rec(n.test))
            add_instruction("je", f"else{control_count}", "")
            
            add_instruction(f"then{control_count}", "", "") 
            for do in n.body: 
                ir_rec(do)
            add_instruction(f"jmp", f"endif{control_count}", "")
            add_instruction(f"else{control_count}", "", "") 
            for welp in n.orelse: 
                ir_rec(welp)
            add_instruction(f"endif{control_count}", "", "")  
        elif isinstance(n, While):
            control_count = cont_count
            curr_while.append(control_count)
            cont_count += 1
            add_instruction(f"while{control_count}", "", "")
            for do in n.body:
                ir_rec(do)
            add_instruction("jmp", f"while{control_count}", "")
            add_instruction(f"endwhile{control_count}", "", "")
        elif isinstance(n, Break):
            add_instruction("jmp", f"endwhile{curr_while.pop()}", "")
        elif isinstance(n, Constant): 
            return f"${n.value}"
        elif isinstance(n, Add): 
            return "addl"
        elif isinstance(n, USub): 
            return "negl"
        elif isinstance(n, Load): 
            pass
        elif isinstance(n, Store): 
            pass
        elif isinstance(n, FunctionDef):
            func_ir = x86()
            arg_string = ", ".join(arg.arg for arg in n.args.args)
            func_ir.add_instruction("Function", arg_string, n.name)
            curr_irs.append(func_ir)
            for stmt in n.body:
                ir_rec(stmt)
            func_irs.append(curr_irs.pop())
        elif isinstance(n, Return):
            add_instruction("movl", ir_rec(n.value), "eax")
            
        else:
            return
    
    ir_rec(tree)
    for func_ir in func_irs:
        for instr in func_ir.body:
            ir.add_instruction(instr['instr'], instr['loc1'], instr['loc2'])
    return ir

def ir_split(ir):
    irs = []
    curr_ir = x86()
    for instr in ir.body:
        if instr['instr'].startswith("Function"):
            if curr_ir.body:
                irs.append(curr_ir)
            curr_ir = x86()
        curr_ir.add_instruction(instr['instr'], instr['loc1'], instr['loc2'])
    if curr_ir.body:
        irs.append(curr_ir)
    return irs

def control_flow(ir):
    ctrl_graph = Ctrl_Graph()
    ctrl_count = 0 
    curr_body = x86()
    ctrl_find = {}
    while_ct = 0
    def connect(cgf):
        for vertex, instr_set in cgf.vertices.items():
            for instr in instr_set.body:
                if instr['instr'].startswith("then"):
                    ctrl_find[instr['instr']] = vertex
                    ctrl_graph.add_edge(vertex, vertex-1)
                elif instr['instr'].startswith("else"):
                    ctrl_find[instr['instr']] = vertex
                    c_to = ctrl_find[f"then{instr['instr'][4:]}"]
                    ctrl_graph.add_edge(vertex, c_to-1)
                elif instr['instr'].startswith("endif"):
                    c_else = vertex-1
                    c_if = ctrl_find[f"else{instr['instr'][5:]}"] - 1
                    ctrl_graph.add_edge(vertex, c_if) 
                    ctrl_graph.add_edge(vertex, c_else)
                elif instr['instr'].startswith("endwhile"):
                    ctrl_graph.add_edge(vertex, vertex-1)

                elif instr['instr'].startswith("while"):
                    ctrl_graph.add_edge(vertex, vertex-1)


            
        
    def new_control():
        nonlocal ctrl_count
        curr_copy = x86()
        curr_copy.body = curr_body.body.copy()
        ctrl_graph.add_vertex(ctrl_count, curr_copy)
        curr_body.body.clear()
        ctrl_count += 1
    for instr in ir.body: 
        if instr['instr'].startswith(("then", "else", "endif")):
            new_control()
            curr_body.body.append(instr)
        elif instr['instr'].startswith(("while", "endwhile")):
            new_control()
            curr_body.body.append(instr)
        elif instr['instr'].startswith("Function"):
            new_control()
            curr_body.body.append(instr)
        else: 
            curr_body.body.append(instr)
    curr_copy = x86()   
    curr_copy.body = curr_body.body.copy()
    ctrl_graph.add_vertex(ctrl_count, curr_copy)
    connect(ctrl_graph)

    return ctrl_graph
                        
def live_cfg(cfg): 
    live_set = {}
    convergent = {}
    while_checks = []
    return_to = []

    def check_convergence(vertex, current_vars): 
        if cfg.vertices[vertex].body[0]['instr'].startswith("then") or cfg.vertices[vertex].body[0]['instr'].startswith("else"):
            if cfg.edges[vertex][0] in convergent:
                return True
            else:
                convergent[cfg.edges[vertex][0]] = current_vars
                return False
    def find_while(endwhile):
        while_ct = endwhile[8:]
        start = 0 
        for vertex in cfg.vertices.keys(): 
            if len(cfg.vertices[vertex].body) > 0:
                if cfg.vertices[vertex].body[-1]['loc1'] == endwhile:
                    start = vertex
        return start
    def do_union(start, end, live_vars):
        for vertex in range(start, end):
            live = live_set[vertex]
            new_live = []
            for life in live:
                new_life = life | live_vars
                new_live.append(new_life)
            live_set[vertex] = new_live
    def do_la_iterative(start_vertex, start_vars):
        stack = [(start_vertex, start_vars)]
        while stack:
            vertex, current_vars = stack.pop()
            if len(cfg.vertices[vertex].body) > 0:
                live = live_analysis(cfg.vertices[vertex], current_vars)
                live_set[vertex] = live
                new_vars = live[0]
                if cfg.vertices[vertex].body[0]['instr'].startswith("endwhile"):
                    start = find_while(cfg.vertices[vertex].body[0]['instr'])
                    return_to.append(vertex)
                    stack.append((start-2, new_vars) ) 
                    stack.append((cfg.edges[vertex][0], set()))
                    continue
                if cfg.vertices[vertex].body[0]['instr'].startswith("while"):
                    do_union(vertex, return_to[-1], new_vars)
                    start = find_while("end" + cfg.vertices[vertex].body[0]['instr'])
                    do_union(vertex, start, while_checks[-1])
                    # Union the while loop with itself
                    do_union(start, return_to[-1], while_checks[-1])
                    return_to.pop()
                    while_checks.pop()
                if cfg.vertices[vertex].body[-1]['loc1'].startswith("endwhile"):
                    triv_vertex,triv_current = stack[-1] 
                    if triv_vertex == vertex-1:  
                        stack.pop()
                        triv_live = live_analysis(cfg.vertices[triv_vertex], triv_current)
                        live_set[triv_vertex] = triv_live
                    while_checks.append(current_vars)
                    continue 
                    
                if cfg.vertices[vertex].body[0]['instr'].startswith(("then", "else")):
                    if check_convergence(vertex, new_vars):
                        converge = convergent[cfg.edges[vertex][0]]
                        del convergent[cfg.edges[vertex][0]]
                        stack.append((cfg.edges[vertex][0], new_vars | converge))
                        continue
                    else: continue
            else:
                live_set[vertex] = []
                new_vars = current_vars

            if len(cfg.edges[vertex]) == 2:
                stack.append((cfg.edges[vertex][0], new_vars))
                stack.append((cfg.edges[vertex][1], new_vars))
            elif len(cfg.edges[vertex]) == 1:
                stack.append((cfg.edges[vertex][0], new_vars))

        last_var = list(cfg.vertices.keys())[-1]
    last_var = list(cfg.vertices.keys())[-1]    
    do_la_iterative(last_var, set())
    return live_set
                
def live_analysis(x86_IR, curr_vars):
    
    def isconst(string):
        return string[0] == "$" or string[0] == "#" 
    live_var = []
    current_vars = curr_vars.copy()
    def build(ir):
        nonlocal current_vars
        if ir['instr'] == "movl":
            loc2 = ir['loc2']
            if loc2 in current_vars:
                current_vars.remove(loc2)
            if not isconst(ir['loc1']):
                current_vars.add(ir['loc1'])
        if ir['instr'] == "addl":
            if not isconst(ir['loc1']):
                current_vars.add(ir['loc1'])
            if not isconst(ir['loc2']):
                current_vars.add(ir['loc2'])
        if ir['instr'] == "negl":
            if not isconst(ir['loc1']):
                current_vars.add(ir['loc1'])
        if ir['instr'] == "call":
            if ir['loc1'] not in ("eval_input") and not isconst(ir['loc1']):
                if ',' in ir['loc1']:
                    _args = ir['loc1'].split(", ")
                    for arg in _args:
                        if not isconst(arg):
                            current_vars.add(arg)

                else:
                    current_vars.add(ir['loc1'])
        if ir['instr'] in ("equals", "cmpl", "nequals"):    
            if not isconst(ir['loc1']): current_vars.add(ir["loc1"])
            if not isconst(ir['loc2']): current_vars.add(ir["loc2"])
        if ir['instr'] == "while":
            if ir['loc1'] in current_vars:
                current_vars.remove(ir['loc1'])
        # print(ir, curr_vars)
            
        
            

    for instrs in reversed(x86_IR.body):
        build(instrs)
        live_var = [current_vars.copy()] + live_var

    # if current_vars:
    #     raise ValueError(f"Error: undefined variables {current_vars}")

    return live_var

def la_flat(la):
    # LA and cfg  holds control blocks this function puts them all back together
    live_analysis = []
    sorted_ctrl = sorted(la.keys())
    for ctrl_la in sorted_ctrl:
        for _set in la[ctrl_la]:
            live_analysis.append(_set)
    live_analysis.append(set())
    return live_analysis

def new_ir(cfg):
    new_ir = x86()
    for vertex in cfg.vertices:
        for instr in cfg.vertices[vertex].body:
            new_ir.add_instruction(instr['instr'], instr['loc1'], instr['loc2'])
    return new_ir
           
def inter_graph(x86_IR, live_vars):
    ig = InterferenceGraph()
    #Array for current alive variables (dyamic for appending)
    shared_vars = []
    def isconst(string):
        return string[0] == "$"
    
    def normalize(edge):
        return tuple(sorted(edge))
    def build_ig(ir, live_vars): 
        nonlocal shared_vars
        # print("\n", ir)
        # print(live_vars, "\n")
        
        if ir['instr'] == "movl":
            
            if not ig.in_vertex(ir['loc2']):
                ig.add_vertex(ir['loc2'], "blank")
                
            #sharing algorithms            
            # shared_with = in_shared_vars(ir['loc2'])
            # if shared_with != None:
            #     shared_vars.remove(shared_with)
                
            if not isconst(ir['loc1']):
                for var in live_vars:
                        ig.add_edge(ir['loc1'], var)
            if ir['loc1'][0] == "#": 
                if not isconst(ir['loc2']):
                    for var in live_vars:
                        ig.add_edge(ir['loc2'], var)
            for var in live_vars:
                ig.add_edge(ir['loc2'], var)
            
                
            
        if ir['instr'] == "addl": 
            if not isconst(ir['loc1']):
                for var in live_vars:
                    if normalize((ir['loc1'], var)) not in shared_vars:
                        ig.add_edge(ir['loc1'], var)
            if not isconst(ir['loc2']): 
                for var in live_vars:
                    ig.add_edge(ir['loc2'], var)
            
        if ir['instr'] == "negl": 
            if not isconst(ir['loc1']):
                for var in live_vars:
                    ig.add_edge(ir['loc1'], var)
        if ir['instr'] in ("equals", "nequals"): 
            if not isconst(ir['loc1']):
                for var in live_vars:
                    ig.add_edge(ir['loc1'], var)
                    ig.add_edge("eax", var)
            if not isconst(ir['loc2']):
                for var in live_vars:
                    ig.add_edge(ir['loc2'], var)
                    ig.add_edge("eax", var)
            
        
        if ir['instr'] == "call": 
            # if ir['loc2'] == 'get_subscript': pdb.set_trace()
            #interference with caller saved registers
            # print(live_vars)
            for var in live_vars: 
                ig.add_edge("eax", var)
                ig.add_edge("ecx", var)
                ig.add_edge("edx", var)

            if ir['loc2'].startswith("callptr"):
                # print("hey", live_vars)
                ig.add_vertex(ir['loc2'], "blank")
                for var in live_vars:
                    ig.add_edge(ir['loc2'], var)

                for var in live_vars: 
                    ig.add_edge("eax", ir['loc2'])
                    ig.add_edge("ecx", ir['loc2'])
                    ig.add_edge("edx",ir['loc2'])
                
            # Handle multiple function calls in ir['loc1']
            if "," in ir['loc1']:
                args = ir['loc1'].split(", ")
                for arg in args:
                    if not isconst(arg):
                        for var in live_vars:
                            ig.add_edge(arg, var)
            # print has a read register (interference)
            elif ir['loc1'] not in ("eval_input") and not isconst(ir['loc1']):
                for var in live_vars:    
                    ig.add_edge(ir['loc1'], var)
        if ir['instr'] == "Function":
            if ir['loc1'] != "":
                args = ir['loc1'].split(", ")
                for arg in args:
                    if not ig.in_vertex(arg):
                        ig.add_vertex(arg, "blank")

    for i, instrs in enumerate(x86_IR.body):
        build_ig(instrs, live_vars[i]) 

    return ig

def get_stack_function(instructions):
    var_to_stack = {}
    stack_offset = 8
    for instr in instructions.body:
        if instr['instr'] == "Function":
            args = instr['loc1'].split(", ")
            
            for arg in args:
                var_to_stack[arg] = f"{stack_offset}(%ebp)"
                stack_offset += 4
            return var_to_stack

    return var_to_stack

def graph_coloring(i_graph, X86_IR, in_stack, nonlocal_stack):
    register_priority = ['apple', 'carrot', 'donut', 'ice_cream', 'banana', 'salad']
    stack_count = -4 + (-4 * len(in_stack)) - (4*nonlocal_stack)
    def get_stack():
        nonlocal stack_count
        open_stack = f'{stack_count}(%ebp)'
        stack_count -= 4
        return open_stack

    def constraints(vertex, edges):
        _constraints = []
        for edge in edges:
            color = i_graph.get_color(edge)
            if color != "blank":
                _constraints.append(color)
        return _constraints

    def color(vertex, shant_eat):
        for yummy in register_priority:
            if yummy not in shant_eat:
                i_graph.change_color(vertex, yummy)
                return
        # spill baby: call get_stack only once and reuse its result
        stack_loc = get_stack()
        i_graph.change_color(vertex, stack_loc)
        in_stack[vertex] = stack_loc

    def color_graph(vertex):
        max_shants = -1
        edges = i_graph.get_edges(vertex)
        favorite = ""
        found = False
        _constraints = constraints(vertex, edges)
        color(vertex, _constraints)
        for edge in edges:
            if i_graph.get_color(edge) == "blank":
                edges_edges = i_graph.get_edges(edge)
                haters = len(constraints(edge, edges_edges))
                if haters > max_shants:
                    max_shants = haters
                    favorite = edge
                    found = True
        if found:
            color_graph(favorite)


    # KEEP THIS IN MIND
    for vertex in in_stack:
        i_graph.change_color(vertex, in_stack[vertex])
    for vertex in i_graph.get_spills():
        color_graph(vertex)

    color_graph(i_graph.highest_order())
    while i_graph.get_first_blank() is not None:
        color_graph(i_graph.get_first_blank())

    return in_stack
       
def spill_code(colored_ig, x86_ir):
    
    vertices = colored_ig.get_vertices()
    
    stacked = [key for key, value in vertices.items() if "ebp" in value]
    #index lol (lots of code to update these two variables (ir_graph changes dynamically when iterated through)
    IR_size = len(x86_ir.body)
    i = 0 
    temp = -1

    run_again = False
    
    def update_size(x): 
        nonlocal IR_size, i
        IR_size = IR_size + x
        i = i + x
    def new_temp(): 
        nonlocal temp
        temp += 1
        return f"sc_temp{temp}"
    
        
    
    def find(ir):
        nonlocal stacked, IR_size, i, run_again
#         if ir['instr'] == "movl":
#             if ir['loc1'] == ir['loc2']:
#                 x86_ir.remove_instruction(i)
#                 update_size(-1)
        
        if ir['instr'] in ("movl", "addl"):
            if ir['loc1'].startswith('$') and ir['loc2'].startswith('$'):
                x86_ir.remove_instruction(i)
                update_size(-1)
        if ir['instr'] == "negl" and ir['loc1'].startswith('$'):
            x86_ir.remove_instruction(i)
            update_size(-1)
        if ir['loc1'] in stacked and ir['loc2'] in stacked:
            # print(ir['loc1'], ir['loc2'])
            check = i-1
            if ir['instr'] == "movl":
                if ir['loc1'] == ir['loc2']: 
                    x86_ir.remove_instruction(i)
                    update_size(-1)
                else:
                    tmp1 =  new_temp()
                    x86_ir.remove_instruction(i)
                    update_size(-1)
                    x86_ir.add_at("movl", ir['loc1'], tmp1, i)
                    update_size(1)
                    x86_ir.add_at("movl", tmp1, ir['loc2'], i)
                    update_size(1)
                run_again = True
            if ir['instr'] == "addl":
                check = i-1
                tmp1 =  new_temp()
                tmp2 = tmp1
                x86_ir.remove_instruction(i)
                update_size(-1)
                x86_ir.add_at("movl", ir['loc1'], tmp1, i)
                update_size(1)
                if ir['loc1'] != ir['loc2']: 
                    tmp2 = new_temp()
                    x86_ir.add_at("movl", ir['loc2'], tmp2, i)
                    update_size(1)
                x86_ir.add_at("addl", tmp1, tmp2, i)
                update_size(1)
                x86_ir.add_at("movl", tmp2, ir['loc2'], i)
                update_size(1)
                # for pls in range(check, i):
                #     print(x86_ir.body[pls])
                run_again = True 
            if ir['instr'] in ("cmpl", "equals", "nequals"):
                tmp1 = new_temp()
                x86_ir.remove_instruction(i)
                update_size(-1)
                x86_ir.add_at("movl", ir['loc1'], tmp1, i)
                update_size(1)
                x86_ir.add_at(ir['instr'], tmp1, ir['loc2'], i)
                update_size(1)
                run_again = True

        else: 
            pass
    while (i < IR_size):
        find(x86_ir.body[i])
        i += 1
        
    return run_again

def get_homes(ir, ig): 
    def ignore(string):
        print(string)
        invalid = ["print", "eval_input", "else", "endif", "while", "endwhile", "project", "inject", "is", "create"]
        for invalid_string in invalid:
            if string.startswith(invalid_string):
                return True
        if string.startswith("Lambda_"):
            return True
        if string == "" or string[0] == "$" or string[0] == "#" :
            return True
        return False
    
    reggie_map = {
    "apple": "eax",
    "carrot": "ecx",
    "donut": "edx",
    "ice_cream": "edi",
    "banana": "ebx",
    "salad": "esi"
    }
    #index
    i = 0
    for instr in ir.body:
        reggie1 = instr['loc1']
        reggie2 = instr['loc2']
        if instr['instr'] == "Function": 
            i += 1  # Increment index and skip processing for "Function"
            continue
        if instr['instr'] == "call":
            if "," in reggie1:
                args = reggie1.split(", ")
                _reggie1 = ""
                for arg in args:
                    if not ignore(arg):
                        color = ig.get_color(arg)
                        if 'ebp' not in color:
                            arg = reggie_map[ig.get_color(arg)]
                        else:
                            arg = ig.get_color(arg)
                    _reggie1 = _reggie1 + arg+  ", "
                reggie1 = _reggie1[:-2]
            elif reggie1: 
                if not ignore(reggie1):
                    color = ig.get_color(reggie1)
                    if 'ebp' not in color:
                        reggie1 = reggie_map[ig.get_color(reggie1)]
                    else:
                        reggie1 = ig.get_color(reggie1)
            if reggie2.startswith("callptr"):
                color = ig.get_color(reggie2)
                if 'ebp' not in color:
                    reggie2 = reggie_map[ig.get_color(reggie2)]
                else:
                    reggie2 = ig.get_color(reggie2)
        elif not ignore(reggie1):
            color = ig.get_color(reggie1)
            if 'ebp' not in color:
                reggie1 = reggie_map[ig.get_color(reggie1)]
            else:
                reggie1 = ig.get_color(reggie1)
        if instr['instr'] != "call" and not ignore(reggie2):
            color = ig.get_color(reggie2)
            if 'ebp' not in color:
                reggie2 = reggie_map[ig.get_color(reggie2)]
            else: 
                reggie2 = ig.get_color(reggie2)
        if ("no color??" in (reggie1, reggie2)):
            print("HEYYY",  instr['loc1'],instr['loc2'] )
        ir.replace_instruction(i, instr['instr'], reggie1, reggie2)
        i += 1

def x86_w_funcs(instructions): 
    code_bodies = []
    curr_function = []
    for instr in instructions.body:
        if instr['instr'] == "Function": 
            curr_x86 = ir_to_x86(curr_function)
            code_bodies.append([curr_x86])
            curr_function = []
        curr_function.append(instr)
    if curr_function:
        curr_x86 = ir_to_x86(curr_function)
        code_bodies.append([curr_x86])
    return code_bodies


def ir_to_x86(instructions):
    def isconst(string):
        return string[0] == "$"
    registers = {"eax", "ebx", "ecx", "edx", "edi", "esi"}
    def get_op(op):
        nonlocal registers
        if op in registers:
            return f"%{op}"
        if op in ("#input", "#val_proj", "#val_inj", "#type_chk", "#list", "#dict", "#subscript") or op.startswith("#ret"):
            return "%eax"
        if op.startswith("Lambda"):
            return f"${op}"
        return op

    def do_compare(instr, operator):
        if instr['loc1'].startswith("$") and instr['loc2'].startswith("$"):
            line = f"    movl {get_op(instr['loc2'])}, %eax \n"
            line += f"    cmpl {get_op(instr['loc1'])}, %eax \n"
        elif instr['loc2'].startswith("$"):
            line = f"    cmpl {get_op(instr['loc2'])}, {get_op(instr['loc1'])} \n"
        else:
            line = f"    cmpl {get_op(instr['loc1'])}, {get_op(instr['loc2'])} \n"
        return line

    def handle_call(instr):
        nonlocal registers
        call = instr['loc2']
        if call in registers or call.endswith("ebp)"):
            call = get_op(call)

        match call:
            case "print": call = "print_any"
            case "eval_input": call = "eval_input_pyobj"
            case "dict_subscript": call = "set_subscript"
            case _: call = call  # Default case to handle other calls
        line = ""
        args = instr['loc1'].split(", ")
        rev_args = args[::-1]  # Reverse the list using slicing
        for arg in rev_args:
            if isconst(arg):
                line += f"    pushl {get_op(arg)}\n"
            else:
                line += f"    movl {get_op(arg)}, %eax \n"
                line += f"    pushl %eax \n"
        line += f"    call {call} \n"
        line += f"    addl ${4 * len(args)}, %esp" 
        return line

    x86 = []
    if instructions.body[0]['instr'] != "Function":
        x86 = ["main"]
    for instr in instructions.body:
        if instr['instr'] in ("movl", "addl"):
            if instr['loc1'] == "#ValueError":
                continue
            if instr['loc1'] in ("#int", "#bool"):
                if instr['loc2'] in {"eax", "ebx", "ecx", "edx", "edi", "esi"}:
                    line = f"    movzbl %al, {get_op(instr['loc2'])}"
                else:
                    line = f"    movzbl %al, %eax \n"
                    line += f"    movl %eax, {get_op(instr['loc2'])}"
            else:
                line = f"    {instr['instr']} {get_op(instr['loc1'])}, {get_op(instr['loc2'])}"
        elif instr['instr'] == "negl":
            line = f"    {instr['instr']} {get_op(instr['loc1'])}"
        elif instr['instr'] == "equals":
            line = do_compare(instr, "sete")
            line += f"    sete %al"
        elif instr['instr'] == "nequals":
            line = do_compare(instr, "setne")
            line += f"    setne %al"
        elif instr['instr'] == "cmpl":
            line = do_compare(instr, "cmpl").rstrip("\n")
        elif instr['instr'] == "je":
            line = f"    je {get_op(instr['loc1'])}"
        elif instr['instr'] == "jmp":
            line = f"    {instr['instr']} {get_op(instr['loc1'])}"
        elif instr['instr'].startswith(("else", "then", "endif", "while", "endwhile")):
            line = f"{instr['instr']}:"
        elif instr['instr'] == "call" and instr['loc1'] not in ("eval_input"):
            line = handle_call(instr)
        elif instr['instr'] == "call" and instr['loc1'] == "eval_input":
            line = f"    call eval_input_pyobj"
        elif instr['instr'] == "Function":
            line = f"{instr['loc2']}"  # Function label
        x86.append(line)

    return "\n".join(x86) + "\n"

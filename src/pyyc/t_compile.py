#!/usr/bin/env python3.10
# THIS FILE IS FOR TESTING PURPOSES ONLY
import ast
from ast import *
import os, sys
import ast
import pprint
# import lark
# from lark import Lark
from back_end.live_analysis import simple_expr_to_x86_ir, live_analysis, inter_graph, graph_coloring, spill_code, get_homes, ir_to_x86, control_flow, live_cfg, la_flat, new_ir, x86_w_funcs, ir_split, get_stack_function


from front_end.flatten import *
from front_end.parse import * 

    
def main_to_x86(count, x86): 
    if not x86: func_name = "main"
    else: 
        func_name = x86.split("\n")[0]
        x86 = x86.split("\n", 1)[1]
    if func_name == "": func_name = "main"
    if func_name == "main":
        add_text = ".section .text\n"
    else: add_text = ""
    asm_main = add_text + (
        f".type {func_name}, @function\n"
        f".globl {func_name}\n"
        f"{func_name}:\n"
        "    pushl %ebp\n"
        "    movl %esp, %ebp\n"
        f"    subl ${count}, %esp\n"
        "    pushl %ebx\n"
        "    pushl %esi\n"
        "    pushl %edi\n\n"
        f"{x86}\n"
        "    popl %edi\n"
        "    popl %esi\n"
        "    popl %ebx\n"
    )
    if func_name == "main":
        asm_main += (
            "    movl $0, %eax\n"
        )
    asm_main += (
        "    movl %ebp, %esp\n"
        "    popl %ebp\n"
        "    ret\n"
    )

    return asm_main


source_code = """
nl = lambda x: (lambda y: x + y)(2)
print(nl(23))

"""
ast_tree = ast.parse(source_code)
ast_tree = unique_valid_PO(ast_tree)
# print(ast.unparse(flat_ast), "\n\n\n")
# print(ast.unparse(flat_ast), "\n\n\n")
ast_tree = uniquify_frees(ast_tree)
all_frees = find_all_frees(ast_tree)
ast_tree = heapify(ast_tree, all_frees)
ast_tree = ast.fix_missing_locations(ast_tree)
ast_tree = closure_conversion(ast_tree, all_frees)
print(ast.unparse(ast_tree), "\n\n\n")
ast_tree = in_func_heapify(ast_tree)
ast.fix_missing_locations(ast_tree)
print(ast.unparse(flatpy_closure(ast_tree)), "\n\n\n")

# At point of heapifying call find_all_frees to get all free vars     
ast_tree = cond_nest(ast_tree)
desugar(ast_tree) 
desugar(ast_tree)  
# print(ast.unparse(ast_tree), "\n\n\n")
flat_ast = flatten(ast_tree)
still_sweet = 1
while still_sweet: 
    still_sweet = desugar(flat_ast)

still_sweet = 1
while still_sweet: 
    still_sweet = desugar(flat_ast)


flat_ast = flat_lists(flat_ast)
flat_ast = flat_dicts(flat_ast)
flat_ast = subscript_remover(flat_ast)
explicated = explicate(flat_ast)
desugar(explicated)
# print("/n",ast.unparse(explicated), "\n\n\n")
# print(ast.unparse(explicated), "\n\n\n")
# # print(ast.dump(explicated, indent = 4))

# # pyobj set_subscript(pyobj c, pyobj key, pyobj val);
flat_w_runtimes = runtime(explicated)


# print(ast.unparse(flat_w_runtimes), "\n\n\n")
ir = simple_expr_to_x86_ir(explicated) 
ir_bodies = ir_split(ir)
x86_bodies = []
final_x86 = "" 
if not ir: 
    final_x86 = main_to_x86(0, "")
for ir in ir_bodies:
    cf = control_flow(ir)
    keep_running = True
    in_stack = get_stack_function(ir)
    # print(in_stack)
    nonlocal_stack = len(in_stack)
    n_ir = ir
    # print(ir)
    while keep_running:
        cf = control_flow(n_ir)
        la = live_cfg(cf)
        flat_la = la_flat(la)
        # pprint.pprint(flat_la)
        n_ir = new_ir(cf)
        keep_running = False
        graph = inter_graph(n_ir, flat_la)
        # print(n_ir)
        # # print("hey", in_stack)
        in_stack = graph_coloring(graph, n_ir, in_stack, nonlocal_stack)
        keep_running = spill_code(graph, n_ir)
    
    # print(graph)
    get_homes(n_ir, graph)
    x86_bodies.append(ir_to_x86(n_ir))
    print(4*len(in_stack)- (4*nonlocal_stack))
    final_x86 += main_to_x86(4*len(in_stack)- (4*nonlocal_stack), ir_to_x86(n_ir)) + "\n\n"
# print(final_x86)



# WORKING ON GRAMMAR
# source_code = """
# x =22 
# print(x and 1)
# """
# print(ast.dump(ast.parse(source_code), indent = 4))
# script_dir = os.path.dirname(os.path.abspath(__file__))
# grammar_path = os.path.join(script_dir, "front_end/p4_grammar.lark")
# with open(grammar_path, "r") as file:
#     grammar = file.read()

# parser = Lark(grammar, parser="lalr", start="module")
# parse_tree = parser.parse(source_code)
# print(parse_tree.pretty())
# # ast_tree = parsetree_to_ast(parse_tree)
#     tmp_1 = get_subscript(s_idx, s_x)
    # ltmp_20 = s_sum
    # ltmp_21 = tmp_1

    # ltmp_21, tmp_1 is not conflicting with s_x


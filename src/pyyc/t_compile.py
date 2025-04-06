#!/usr/bin/env python3.10
# THIS FILE IS FOR TESTING PURPOSES ONLY
import ast
from ast import *
import os, sys
import ast
import pprint
# import lark
# from lark import Lark
from back_end.live_analysis import simple_expr_to_x86_ir, live_analysis, inter_graph, graph_coloring, spill_code, get_homes, ir_to_x86, control_flow, live_cfg, la_flat, new_ir


from front_end.flatten import *
from front_end.parse import * 

    
def main_to_x86(count, x86): 
    # vars_dict = stack_vars(main_ast)
    asm_main = (
        ".globl main\n"
        "main:\n"
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
        "    movl $0, %eax\n"
        "    movl %ebp, %esp\n"
        "    popl %ebp\n"
        "    ret\n"
    )
    return asm_main


source_code = """
def is_even(n):
    return is_odd(n + -1) if n != 0 else True

is_odd = lambda n: is_even(n + -1) if n != 0 else False


print(is_odd(23))


"""
ast_tree = ast.parse(source_code)
print(ast.dump(ast_tree, indent = 4))
ast_tree = unique_valid_PO(ast_tree)
ast_tree = cond_nest(ast_tree)
desugar(ast_tree)   
flat_ast = flatten(ast_tree)
still_sweet = 1
while still_sweet: 
    still_sweet = desugar(flat_ast)
flat_ast = flatten(ast_tree)
print()
print(ast.unparse(flat_ast))
ast_tree = uniquify_frees(ast_tree)
all_frees = find_all_frees(ast_tree)
print(all_frees)
ast_tree = heapify(ast_tree, all_frees)
ast_tree = closure_conversion(ast_tree)
ast.fix_missing_locations(ast_tree)
print("\n\n\n",ast.unparse(flatpy_closure(ast_tree)))



# At point of heapifying call find_all_frees to get all free vars     

still_sweet = 1
while still_sweet: 
    still_sweet = desugar(flat_ast)

# flat_ast = flat_lists(flat_ast)
# flat_ast = flat_dicts(flat_ast)
# print(ast.unparse(flat_ast), "\n\n\n")
# explicated = explicate(flat_ast)
# # desugar(explicated)
# print(ast.unparse(explicated))
# # pyobj set_subscript(pyobj c, pyobj key, pyobj val);
# # flat_w_runtimes = runtime(explicated)
# ir = simple_expr_to_x86_ir(explicated) 
# cf = control_flow(ir)
# # print(cf)
# # # # # # # opt_cf = lvn(cf)
# # # # # # # opt_cf = find_dead(opt_cf)
# # # # # # # pprint.pprint(opt_cf)
# # live = live_cfg(cf)
# # # flat_la = la_flat(live)
# keep_running = True
# in_stack = {}
# n_ir = ir
# print(ir)
# while keep_running:
#     cf = control_flow(n_ir)
#     la = live_cfg(cf)
#     flat_la = la_flat(la)
#     # pprint.pprint(flat_la)
#     n_ir = new_ir(cf)
#     graph = inter_graph(n_ir, flat_la)
#     in_stack = graph_coloring(graph, n_ir, in_stack)
#     keep_running = spill_code(graph, n_ir)
# # print("\n", n_ir)
# get_homes(n_ir, graph)
# # # print(ir_to_x86(ir))
# x86 = ir_to_x86(n_ir)
# # #Flat-to x-86
# asm_code = main_to_x86(4*len(in_stack), x86)
# print("\n\n\n", asm_code[:10000 ])



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


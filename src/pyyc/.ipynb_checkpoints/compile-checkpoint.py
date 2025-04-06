#!/usr/bin/env python3.10
import ast
from ast import *
import os, sys
import ast
import pprint
# import lark
# from lark import Lark
from live_analysis import x86, InterferenceGraph, simple_expr_to_x86_ir, live_analysis, inter_graph, graph_coloring, spill_code, get_homes, ir_to_x86, control_flow, live_cfg, la_flat, lvn 
from flatten import *
from parse import * 
def unique_valid_PO(tree):
    class Uniquify(ast.NodeTransformer):
        def visit_Name(self, node):
            if (node.id not in ("print", "id", "eval", "input", "int" )):
                node.id = f"s_{node.id}"
            return node

    transformer = Uniquify()
    unique = transformer.visit(tree)
    return unique
    

        
    

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

def main():
    # script_dir = os.path.dirname(os.path.abspath(__file__))
#     grammar_path = os.path.join(script_dir, "p0_grammar.lark")
#     with open(grammar_path, "r") as file:
#          grammar = file.read()

#     parser = Lark(grammar, parser="lalr", start="module")

    input_file = sys.argv[1]

    output_file = f"{os.path.splitext(input_file)[0]}.s"

    try:
        with open(input_file, 'r') as f:
            source_code = f.read()

        # Pre-Compiler
        ast_tree = ast.parse(source_code)
        ast_tree = unique_valid_PO(ast_tree)
        ast_tree = cond_nest(ast_tree)
        desugar(ast_tree)                        
        flat_ast = flatten(ast_tree)
        still_sweet = 1
        while still_sweet: 
            still_sweet = desugar(ast_tree)

        ir = simple_expr_to_x86_ir(flat_ast) 
        keep_running = True
        in_stack = {}
        while keep_running:
            cf = control_flow(ir)
            opt_cf = lvn(cf)
            la = live_cfg(cf)
            flat_la = la_flat(la)
            graph = inter_graph(ir, flat_la)
            print(graph)
            in_stack = graph_coloring(graph, ir, in_stack)
            keep_running = spill_code(graph, ir)

        get_homes(ir, graph)
        x86 = ir_to_x86(ir)
        #Flat-to x-86
        asm_code = main_to_x86(4*len(in_stack), x86)
        
        with open(output_file, 'w') as f:
            f.write(asm_code)
    except Exception as e:
        sys.exit(1)

if __name__ == '__main__':
    main()
    



# source_code = """
# x = 1
# y = 2
# z = 3
# w = 4
# v = x + y + z + w
# print((x and y) and (z and w))
# v = v + v + v + v
# print(v and v)

# """
# print(ast.dump(ast.parse(source_code), indent = 4))
# ast_tree = ast.parse(source_code)
# ast_tree = unique_valid_PO(ast_tree)
# ast_tree = cond_nest(ast_tree)
# desugar(ast_tree)                      
# flat_ast = flatten(ast_tree) 
# still_sweet = 1
# while still_sweet: 
#     still_sweet = desugar(ast_tree)
# print(ast.unparse(flat_ast)) 
# ir = simple_expr_to_x86_ir(flat_ast) 
# cf = control_flow(ir)
# opt_cf = lvn(cf)
# pprint.pprint(opt_cf)
# live = live_cfg(cf)
# flat_la = la_flat(live)
# keep_running = True
# in_stack = {}
# while keep_running:
#     cf = control_flow(ir)
#     opt_cf = lvn(cf)
#     la = live_cfg(cf)
#     flat_la = la_flat(la)
#     graph = inter_graph(ir, flat_la)
#     print(graph)
#     in_stack = graph_coloring(graph, ir, in_stack)
#     keep_running = spill_code(graph, ir)

# get_homes(ir, graph)
# # print(graph)
# # print(ir_to_x86(ir))
# x86 = ir_to_x86(ir)
# #Flat-to x-86
# asm_code = main_to_x86(4*len(in_stack), x86)
# print("\n\n\n", asm_code)

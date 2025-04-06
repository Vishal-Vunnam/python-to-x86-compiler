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

def main():
    # script_dir = os.path.dirname(os.path.abspath(__file__))
#     grammar_path = os.path.join(script_dir, "p0_grammar.lark")
#     with open(grammar_path, "r") as file:
#          grammar = file.read()

#     parser = Lark(grammar, parser="lalr", start="module")

    input_file = sys.argv[1]
    output_flatpy = f"{os.path.splitext(input_file)[0]}.flatpy"

    output_file = f"{os.path.splitext(input_file)[0]}.s"

    try:
        with open(input_file, 'r') as f:
            source_code = f.read()  
        
        ast_tree = ast.parse(source_code) 
        ast_tree = unique_valid_PO(ast_tree)
        ast_tree = cond_nest(ast_tree)
        desugar(ast_tree)   
        flat_ast = flatten(ast_tree)
        still_sweet = 1
        while still_sweet: 
            still_sweet = desugar(flat_ast)
        flat_ast = flatten(ast_tree)
        ast_tree = uniquify_frees(ast_tree)
        all_frees = find_all_frees(ast_tree)
        ast_tree = heapify(ast_tree, all_frees)
        ast_tree = closure_conversion(ast_tree)
        ast.fix_missing_locations(ast_tree)
        with open (output_flatpy, 'w') as f:
            f.write(ast.unparse(flatpy_closure(ast_tree)))
        flat_ast = flat_lists(flat_ast)
        flat_ast = flat_dicts(flat_ast)
        explicated = explicate(flat_ast)
        ast.fix_missing_locations(explicated)
        flat_w_runtimes = runtime(explicated)

        ir = simple_expr_to_x86_ir(explicated) 
        keep_running = True
        in_stack = {}
        n_ir = ir

        while keep_running:
            cf = control_flow(n_ir)
            la = live_cfg(cf)
            flat_la = la_flat(la)

            n_ir = new_ir(cf)
            graph = inter_graph(n_ir, flat_la)
            in_stack = graph_coloring(graph, n_ir, in_stack)
            keep_running = spill_code(graph, n_ir)

        get_homes(n_ir, graph)
        x86 = ir_to_x86(n_ir)
        asm_code = main_to_x86(4*len(in_stack), x86)
        with open(output_file, 'w') as f:
            f.write(asm_code)
    except Exception as e:
        sys.exit(1)

if __name__ == '__main__':
    main()
    print("running")
    






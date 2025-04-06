
# from lark import Lark
# import ast
# def parsetree_to_ast(larktree) -> ast.Module:
#     class ToAst(lark.Transformer):
#         def module(self, items):
#             return ast.Module(body=items)


#         def expression(self, items):
#             print(ast.dump(items[0], indent = 4))
#             return items[0]

#         def print_call(self, items):
#             expression = items[0]
#             return ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
#                                            args=[expression]))

#         def stmt(self, items):
#             return ast.Expr(items[0])

#         def assign(self, items):
#             var_name, expression = items
#             return ast.Assign(targets=[var_name], value=expression)

#         def do_unary(self, items):
#             return ast.UnaryOp(op=items[0], operand=items[1])

#         def do_binary(self, items):
#             return ast.BinOp(left=items[0], op=items[1], right=items[2])

#         def parens(self, items):
#             return items[0]

#         def user_input(self, items):
#             """i.e. `eval(input())`"""
#             return ast.Call(func=ast.Name(id='eval', ctx=ast.Load()),
#                             args=[
#                                 ast.Call(func=Name(id='input'), args=[])
#                             ])


#         def NUMBER(self, token):
#             return ast.Constant(value=int(token))

#         def CNAME(self, token):
#             return ast.Name(id=token.value, ctx=ast.Load())

#         def add(self, items):
#             return ast.Add()

#         def usub(self, items):
#             return ast.USub()
#     return ToAst().transform(larktree)

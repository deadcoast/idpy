
grammar = """
    start: expr
    expr: atom
        | expr "+" atom   -> add
        | expr "-" atom   -> sub
        | expr "*" atom   -> mul
        | expr "/" atom   -> div
        | expr "^" atom   -> pow
    atom: NUMBER         -> number
         | "(" expr ")"
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

class CalculateTreeMethod(Transformer):
    def __init__(self):
        self.parser = Lark(grammar, parser='lalr', transformer=self)
        self.result = None

    def __call__(self, args: List[Union[int, float]]) -> Union[int, float]:
        method_name = self.parser.parse_args().method_name
        if method_name == 'add':
            return self.add(args)
        elif method_name == 'sub':
            return self.sub(args)
        elif method_name == 'mul':
            return self.mul(args)
        elif method_name == 'div':
            return self.div(args)
        elif method_name == 'pow':
            return self.pow(args)
        else:
            raise ValueError(f"Invalid method name: {method_name}")

    def add(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] + args[1]

    def sub(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] - args[1]

    def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] * args[1]

    def pow(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] ** args[1]

    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] / args[1]

    def number(self, args: List[Union[int, float]]) -> float:
        try:
            return float(args[0])
        except TypeError as e:
            raise TypeError("Invalid number type") from e
        except ValueError as e:
            raise ValueError("Invalid number format") from e


parser = Lark(grammar, parser='lalr', transformer=CalculateTreeMethod())
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11

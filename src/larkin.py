import wrapt
import sys
from functools import wraps
from typing import List, Union
import prompt
import os

import argparse
from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = argparse.ArgumentParser(description='Lark CLI for handling file imports and exports.')
parser.add_argument('--import-dir', type=str, help='Path to the import directory')
parser.add_argument('--export-dir', type=str, help='Path to the export directory')
args = parser.parse_args()
input_dir = args.import_dir

output_dir = args.export_dir,
if not os.path.exists(input_dir):
    print(f"Directory {input_dir} does not exist")
    exit(1)

export_directory = "/ path / to / export"

if not os.path.exists(export_directory):
    os.makedirs(export_directory)
    
parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11



def get_directory_path(directory_path: str) -> str:
    """
    Get the directory path based on the given parameter.

    Args:
        directory_path (str): The parameter used to determine the directory path.

    Returns:
        str: The directory path if it exists, otherwise a default value or a meaningful message.

    """
    if not isinstance(directory_path, str):
        raise ValueError("Invalid directory path: directory_path must be a string")
    try:
        if not os.path.isdir(directory_path):
            return "Default directory path"
        if not os.access(directory_path, os.R_OK):
            return "Error: Directory is not readable"
        return directory_path
    except ValueError as e:
        return "Error: Invalid directory path"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Default value or meaningful message"


input_dir = get_directory_path("Enter the path to the import directory: ")

output_dir = get_directory_path("Enter the path to the export directory: ")

if input_dir == "Default directory path" or output_dir == "Default directory path":
    print("Invalid directory path provided. Exiting...")
    exit(1)

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
              | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

grammar = """
?start: expression
?expression: term
           | expression "+" term   -> add
           | expression "-" term   -> sub
?term: factor
      | term "*" factor  -> mul
      | term "/" factor  -> div
?factor: NUMBER           -> number
       | "(" expression ")"
NUMBER: /[0-9]+/
%import common.WS
%ignore WS
"""

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer

"""
NUMBER: /[0-9]+/ %import common.WS %ignore WS """

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11](<from lark import Lark, Transformer


class InvalidArgumentsError(Exception):
    """
    Exception raised for invalid arguments.

    This exception is raised when invalid arguments are passed to a function or method.

    Attributes:
    message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return "Invalid arguments provided."

    def __repr__(self):
        return f"InvalidArgumentsError()"

    def log_error(self):
        # code to log the error
        pass

class CalculateTree(Transformer):
    def validate_args(func):
        @wrapt.decorator
        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments
        return wrapper

    @validate_args
    def add(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args
        result = arg1 + arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Sum exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Sum exceeds maximum limit of float")
        min_limit = min(int, float)
        if result < min_limit:
            raise ValueError("Error: Result is below the minimum limit of int or float")
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands")  # Include a check for operands not being numbers
    @validate_args
    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        try:
            return args[0] / args[1]
        except ZeroDivisionError as e:
            raise ValueError("Error: Division by zero") from e
        operand1, operand2 = args
    def number(self, args: List[Union[int, float]]) -> int:
        try:
            return int(args[0])
        except TypeError as e:
            raise TypeError("Invalid number type") from e
        except ValueError as e:
            raise ValueError("Invalid number format") from e
        except TypeError as e:
            raise ValueError("Error: Invalid operands") from e

    @validate_args
    def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args  # Suggestion 1: Unpack the list into two variables for better readability
        result = arg1 * arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Multiplication result exceeds maximum limit of int")  # Suggestion 2: Include a check for multiplication result exceeding maximum limit of int or float
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Multiplication result exceeds maximum limit of float")
        if isinstance(result, (int, float)):
            min_limit = min(int, float)
            if result < min_limit:
                raise ValueError("Error: Result is below the minimum limit of int or float")  # Suggestion 3: Include a check for multiplication result below minimum limit of int or float
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands")  # Suggestion 4: Include a check for operands not being numbers
        if None in args:
            raise ValueError("Error: Operands cannot be None")  # Suggestion 5: Include a check for operands being None
        return result


    def validate_args(args): # Suggestion 6: Add type hints to the function signature
        if len(args) != 2:
            raise ValueError("Expected exactly 2 arguments")
        for arg in args:
            if not isinstance(arg, (int, float)):
                raise TypeError("Invalid operands")
        return True

    @validate_args
    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        try:
            return args[0] / args[1]
        except ZeroDivisionError as e:
            raise ValueError("Error: Division by zero")(e)
        min_limit = min(int, float)
        if result < min_limit:
            raise ValueError("Error: Result is below the minimum limit of int or float")
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands")
        return result

    def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args
        result = arg1 * arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Multiplication result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Multiplication result exceeds maximum limit of float")
        min_limit = min(int, float)
        if result < min_limit:
            raise ValueError("Error: Result is below the minimum limit of int or float")
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands")
        return result

    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        try:
            return args[0] / args[1]
        except ZeroDivisionError as e:
            raise ValueError("Error: Division by zero") from e
        operand1, operand2 = args
        result = operand1 / operand2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Division result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Division result exceeds maximum limit of float")
        min_limit = min(int, float)
        if result < min_limit:
            raise ValueError("Error: Result is below the minimum limit of int or float")
        if operand1 is None or operand2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(operand1, (int, float)) or not isinstance(operand2, (int, float)):
            raise TypeError("Invalid operands")
        return result

    def number(self, args: List[Union[int, float]]) -> int:
        try:
            return int(args[0])
        except TypeError as e:
            raise TypeError("Invalid number type") from e
        except ValueError as e:
            raise ValueError("Invalid number format")(e) from e

    def __call__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0]

    def __add__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.add(args)

    def __sub__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.sub(args)

    def __mul__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.mul(args)

    def __truediv__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.div(args)

    def __pow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.pow(args)

    def __rpow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.pow(args)

    def __radd__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.add(args)

    def __rsub__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.sub(args)

    def __rmul__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.mul(args)

    def __rtruediv__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.div(args)

    def __neg__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return -args[0]

    def __pos__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return +args[0]

    def __invert__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return ~args[0]

    def __abs__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return abs(args[0])

    def __round__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return round(args[0])

    def __floor__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return math.floor(args[0])

    def __ceil__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return math.ceil(args[0])

    def __int__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return int(args[0])

    def __float__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return float(args[0])

    def __str__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return str(args[0])

    def __repr__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return repr(args[0])

    def __eq__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] == args[1]

    def __ne__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] != args[1]

    def __lt__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] < args[1]

    def __gt__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] > args[1]

    def __le__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] <= args[1]

    def __ge__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] >= args[1]

    def add(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] + args[1]

    def sub(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] - args[1]

    def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] * args[1]

    def pow(self, args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] ** args[1]

    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        try:
            return args[0] / args[1]
        except ZeroDivisionError as e:
            raise ValueError("Error: Division by zero") from e

    def number(self, args: List[Union[int, float]]) -> int:
        try:
            return int(args[0])
        except TypeError as e:
            raise TypeError("Invalid number type") from e
        except ValueError as e:
            raise ValueError("Invalid number format")(e) from e



    def __pow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.pow(args)

    def __rpow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.pow(args)

    def __truediv__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.div(args)

    def __rtruediv__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return self.div(args)

    def __neg__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return -args[0]

    def __pos__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return +args[0]

    def __invert__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return ~args[0]

    @validate_args
    def calculate(args: List[Union[int, float]]) -> Union[int, float]:
        return args[0]

    @validate_args
    def add(args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] + args[1]

    @validate_args
    def sub(args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] - args[1]

    @validate_args
    def mul(args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] * args[1]

    @validate_args
    def div(args: List[Union[int, float]]) -> Union[int, float]:
        try:
            return args[0] / args[1]
        except ZeroDivisionError as e:
            raise ValueError("Error: Division by zero") from e

    @validate_args
    def pow(args: List[Union[int, float]]) -> Union[int, float]:
        return args[0] ** args[1]

    @validate_args
    def number(args: List[Union[int, float]]) -> int:
        try:
            return int(args[0])
        except TypeError as e:
            raise TypeError("Invalid number type") from e
        except ValueError as e:
            raise ValueError("Invalid number format")(e) from e


class CalculateTree(Transformer):
def __init__(self):
    self.result = None

def __call__(self, args: List[Union[int, float]]) -> Union[int, float]:
    return args[0]

def add(self, args: List[Union[int, float]]) -> Union[int, float]:
    return args[0] + args[1]

def sub(self, args: List[Union[int, float]]) -> Union[int, float]:
    return args[0] - args[1]

def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
    return args[0] * args[1]

def pow(self, args: List[Union[int, float]]) -> Union[int, float]:
    return args[0] ** args[1]
def div(self, args: List[Union[int, float]]) -> Union[int, float]:
    try:
        return args[0] / args[1]
    except ZeroDivisionError as e:
        raise ValueError("Error: Division by zero") from e

def number(self, args: List[Union[int, float]]) -> int:
    try:
        return int(args[0])
    except TypeError as e:
        raise TypeError("Invalid number type") from e
    except ValueError as e:
        raise ValueError("Invalid number format")(e) from e


parser = Lark(grammar, parser='lalr', transformer=CalculateTree())
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11

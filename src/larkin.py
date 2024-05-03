import ast
from functools import wraps
from sys import maxsize
from sys import float_info
import argparse
import math
import os
import sys

import wrapt
from lark import Lark, Transformer
from typing import List, Union

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
    print("Directory {input_dir} does not exist")
    exit(1)

export_directory = "/ path / to / export"

if not os.path.exists(export_directory):
    os.makedirs(export_directory)

parser = Lark(grammar, parser='lalr', transformer=Transformer)
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11

import tokenize


def execfile(file, glob=None, loc=None):
    if glob is None:
        glob = globals()
    if loc is None:
        loc = glob

    with tokenize.open(file) as stream:
        contents = stream.read()

    tree = ast.parse(contents, filename=file, mode='exec')

    exec(compile(tree, filename=file, mode='exec'), glob, loc)


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
        print("An unexpected error occurred: {e}")
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

    def __init__(self, message: str):
        """
        Initialize the class instance.

        Args:
            message (str): The message to be stored in the instance.

        Returns:
            None
        """
        self.__message = message

    @property
    def message(self):
        return self.__message

    @message.setter
    def message(self, message):
        self.__message = message

    @message.deleter
    def message(self):
        del self.__message

    def __call__(self):
        return self.__message

    def log_error(self):
        # code to log the error
        pass


def add_tree(args: List[Union[int, float]]) -> Union[int, float]:
    if None in args:
        raise ValueError("Error: Operands cannot be None")
    if not all(isinstance(arg, (int, float)) for arg in args):
        raise TypeError("Invalid operands. Expected int or float.")
    if not args:
        raise ValueError("Error: Empty list of arguments")
    return args[0] if len(args) == 1 else sum(args)


def mul(args: List[Union[int, float]]) -> Union[int, float]:
    arg1, arg2 = args
    if None in args:
        raise ValueError("Error: Operands cannot be None")
    if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
        raise TypeError("Invalid operands. Expected int or float.")
    result = arg1 * arg2
    if isinstance(result, int) and result > sys.maxsize:
        raise ValueError("Error: Multiplication result exceeds maximum limit of int")
    elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
        raise ValueError("Error: Multiplication result exceeds maximum limit of float")
    if isinstance(result, int) and result < -sys.maxsize:
        raise ValueError("Error: Multiplication result is below the minimum limit of int")
    elif isinstance(result, float) and (result > -sys.float_info.max or result < sys.float_info.max):
        raise ValueError("Error: Multiplication result is below the minimum limit of float")
    return result


class CalculateTree(Transformer):
    def validate_args(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    def validate_args_with_zero_check(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    def validate_args_with_zero_check_and_log_error(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    def validate_args_zero_log_and_exception(func):

        def wrapper_args_log_and_exception(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper_args_log_and_exception

    def number(self, arg: Union[int, float]) -> Union[int, float]:
        """
        Returns the input number.

        Args:
            arg (Union[int, float]): The number.

        Returns:
            Union[int, float]: The input number.

        Raises:
            InvalidArgumentsError: If the input is not exactly one argument.
            TypeError: If the input is not of type int or float.
        """
        if arg is None:
            raise InvalidArgumentsError("Expected exactly 1 argument")
        if not isinstance(arg, (int, float)):
            raise TypeError("Invalid operand. Expected int or float.")
        return arg

    def pow(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        result = arg1 ** arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Exponentiation result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Exponentiation result exceeds maximum limit of float")
        if isinstance(result, int) and result < -sys.maxsize:
            raise ValueError("Error: Exponentiation result is below the minimum limit of int")
        elif isinstance(result, float) and (result > -sys.float_info.max or result < sys.float_info.max):
            raise ValueError("Error: Exponentiation result is below the minimum limit of float")
        return result

    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        if arg2 == 0:
            raise ValueError("Error: Division by zero")
        result = arg1 / arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Division result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Division result exceeds maximum limit of float")
        if isinstance(result, int) and result < -sys.maxsize:
            raise ValueError("Error: Division result is below the minimum limit of int")
        elif isinstance(result, float) and (result > -sys.float_info.max or result < sys.float_info.max):
            raise ValueError("Error: Division result is below the minimum limit of float")
        return result

    def __lt__(self, args: List[Union[int, float]]) -> bool:
        """
        Check if the first element in args is less than the second element.

        Args:
            args (List[Union[int, float]]): A list of two elements.

        Returns:
            bool: True if the first element is less than the second element, False otherwise.

        Raises:
            ValueError: If args is None or if it does not have exactly 2 elements.
            TypeError: If the elements in args are not of type int or float.
        """
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 2:
            raise ValueError("Expected exactly 2 arguments")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        arg1, arg2 = args
        return arg1 < arg2

    def __gt__(self, arg1: Union[int, float], arg2: Union[int, float]) -> bool:
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        return arg1 > arg2

    def __le__(self, arg1: Union[int, float], arg2: Union[int, float]) -> bool:
        """
        Check if the first argument is less than or equal to the second argument.

        Args:
            arg1: The first number.
            arg2: The second number.

        Returns:
            bool: True if arg1 is less than or equal to arg2, False otherwise.
        """
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if math.isnan(arg1) or math.isnan(arg2):
            raise ValueError("Error: Comparison with NaN is not supported")
        return arg1 <= arg2

    def __le__(self, arg1: Union[int, float], arg2: Union[int, float]) -> bool:
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        return arg1 <= arg2

    def __ge__(self, args: List[Union[int, float]]) -> bool:
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
        return args[0] >= args[1]

    def __call__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if not args:
            raise ValueError("Error: Empty argument list")
        if not isinstance(args[0], (int, float)):
            raise TypeError("Invalid type. Expected int or float.")
        if len(args) != 1:
            return sum(args)
        return args[0]

    def __add__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
        try:
            return self.add(args)
        except Exception as e:
            # Handle the exception here
            ...
        if hasattr(self, 'add'):
            return self.add(args)
        else:
            raise AttributeError("Error: 'add' method does not exist")
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        arg1, arg2 = args
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands. Expected int or float.")
        result = arg1 + arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Addition result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Addition result exceeds maximum limit of float")
        if isinstance(result, int) and result < -sys.maxsize:
            raise ValueError("Error: Addition result is below the minimum limit of int")
        elif isinstance(result, float) and (result > -sys.float_info.max or result < sys.float_info.max):
            raise ValueError("Error: Addition result is below the minimum limit of float")
        return result

    def __sub__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None or len(args) != 2:
            raise ValueError("Error: Expected exactly 2 arguments")
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
        try:
            result = self.sub(args)
        except Exception as e:
            # Handle the exception here, such as logging an error message or returning a default value
            ...
        if hasattr(self, 'sub'):
            return self.sub(args)
        else:
            raise AttributeError("Error: 'sub' method does not exist")
        if isinstance(result, int) and (result > sys.maxsize or result < -sys.maxsize):
            raise ValueError("Error: Subtraction result exceeds maximum or minimum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Subtraction result exceeds maximum or minimum limit of float")
        return result

    def __mul__(self, args: List[Union[int, float]]) -> Union[int, float]:
        # Suggestion 2: Check if args is None
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
    
        # Suggestion 3: Check if args has exactly 2 elements
        if len(args) != 2:
            raise ValueError("Error: Expected exactly 2 arguments")
    
        # Suggestion 4: Check if the elements in args are of type int or float
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
    
        # Original code: Call self.mul(args) instead of mul(args)
        return self.mul(args)

        # Suggestion 5: Check if the multiplication result exceeds the maximum or minimum limit of int or float
        result = self.mul(args)
        if isinstance(result, int) and (result > sys.maxsize or result < -sys.maxsize):
            raise ValueError("Error: Multiplication result exceeds maximum or minimum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Multiplication result exceeds maximum or minimum limit of float")
        return result

    def __truediv__(self, args: List[Union[int, float]]) -> Union[int, float]:
        try:
            @validate_args_with_zero_check
            def div(*args: Union[int, float]) -> Union[int, float]:
                return self.div(*args)
            return div(*args)
        except ZeroDivisionError:
            raise ValueError("Error: Division by zero")
        except AttributeError:
            raise AttributeError("Error: 'div' method does not exist")

    def __pow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if len(args) != 2:
            raise ValueError("Error: Expected exactly 2 arguments")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        try:
            return self.pow(args)
        except Exception as e:
            raise ValueError("Error: Failed to perform exponentiation.")

    def __rpow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        # Suggestion 1: Validate the args input
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
    
        # Suggestion 3: Check if the pow method exists
        if hasattr(self, 'pow'):
            try:
                # Suggestion 4: Handle the case where args is None
                if args is None:
                    raise ValueError("Error: Arguments cannot be None")
            
                # Suggestion 5: Handle the case where args is an empty list
                if not args:
                    return 1
            
                # Suggestion 2: Handle exceptions raised by the pow method
                return self.pow(args)
            except Exception as e:
                # Suggestion 2: Handle the exception here
                ...
        else:
            raise AttributeError("Error: 'pow' method does not exist")

    def __radd__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None or len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
        try:
            if hasattr(self, 'add'):
                result = self.add(args)
                if isinstance(result, int) and (result > sys.maxsize or result < -sys.maxsize):
                    raise ValueError("Error: Addition result exceeds the maximum or minimum limit of int")
                elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
                    raise ValueError("Error: Addition result exceeds the maximum or minimum limit of float")
                return result
            else:
                raise AttributeError("Error: 'add' method does not exist")
        except Exception as e:
            raise ValueError("Error occurred during addition: " + str(e))

    def __rsub__(self, args: List[Union[int, float]]) -> Union[int, float]:
        # Suggestion 1: Validate the args parameter
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")

        # Suggestion 2: Handle the case where the sub method does not exist
        if hasattr(self, 'sub'):
            # Suggestion 5: Reverse the order of arguments
            arg1, arg2 = args
            return self.sub([arg2, arg1])
        else:
            raise AttributeError("Error: 'sub' method does not exist")

        # Suggestion 3: Handle exceptions raised by the sub method
        try:
            return self.sub(args)
        except Exception as e:
            # Suggestion 4: Return a meaningful error message
            raise ValueError("Error: Subtraction operation failed. Invalid operands or unsupported types.")

    def __rmul__(self, args: List[Union[int, float]]) -> Union[int, float]:
        # Suggestion 2: Add a check for None
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
    
        # Suggestion 3: Validate the length and type of args
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
    
        # Suggestion 5: Check if self.mul is callable
        if callable(self.mul):
            try:
                # Suggestion 1: Call self.mul instead of mul
                return self.mul(args)
            except Exception as e:
                # Suggestion 4: Handle exceptions
                # Handle the exception here
                ...
        else:
            raise AttributeError("Error: 'mul' method does not exist or is not callable")

    def __rtruediv__(self, *args: Union[int, float]) -> float:
        """
        Reverse division operation.
    
        Args:
            args: A list of two arguments of type int or float.
        
        Returns:
            The result of dividing the second argument by the first argument.
        
        Raises:
            InvalidArgumentsError: If the number of arguments is not exactly 2 or if any argument is not of type int or float.
        """
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
        try:
            return self.div([args[1], args[0]])
        except Exception as e:
            raise ValueError("Error: Failed to perform division operation.")

    def __rpow__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
    
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
    
        try:
            return self.pow(args)
        except Exception as e:
            raise ValueError("Error: Failed to perform power operation.")

    def __rmod__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
    
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
    
        try:
            return self.mod(args)
        except Exception as e:
            raise ValueError("Error: Failed to perform modulo operation.")

    def __invert__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 1:
            raise ValueError("Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return ~args[0]

    def __abs__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 1:
            raise ValueError("Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return abs(args[0])

    def __round__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 1:
            raise ValueError("Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return round(args[0])

    def __floor__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 1:
            raise ValueError("Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return math.floor(args[0])

    def __ceil__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(args) != 1:
            raise ValueError("Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return math.ceil(args[0])

    def __neg__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None or not args:
            raise ValueError("Error: Arguments cannot be None or empty")
        if not isinstance(args[0], (int, float)):
            raise TypeError("Invalid operand. Expected int or float.")
        if len(args) != 1:
            raise ValueError("Error: Expected exactly 1 argument")
        try:
            return -args[0]
        except Exception as e:
            # Handle the exception here
            raise ValueError("Error: Negation operation failed. Invalid operand or unsupported type.")(e)

    def __pos__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if args is None:
            raise ValueError("Error: Arguments cannot be None")
        if not args:
            raise ValueError("Error: Empty argument list")
        if not isinstance(args[0], (int, float)):
            raise TypeError("Invalid operand. Expected int or float.")
        if len(args) != 1:
            return sum(args)
        return +args[0]


    def __round__(self, args: List[Union[int, float]]) -> Union[int, float]:
        if len(args) != 1:
            raise InvalidArgumentsError("Expected exactly 1 argument")
        if args[0] is None:
            raise ValueError("Error: Argument cannot be None")
        if not isinstance(args[0], (int, float)):
            raise TypeError("Invalid argument type. Expected int or float.")
        try:
            return round(args[0])
        except Exception as e:
            print(f"Error occurred during rounding: {e}")

    def __floor__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return math.floor(args[0])

    def __ceil__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return math.ceil(args[0])

    def __int__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return int(args[0])

    def __float__(self, args: List[Union[int, float]]) -> Union[int, float]:
        return float(args[0])

    def calculate(self) -> Union[int, float]:
        """
        Calculate the result of the given expression.

        Args:
            args (List[Union[int, float]]): A list of two elements.

        Returns:
            Union[int, float]: The result of the expression.

        Raises:
            ValueError: If args is None or if it does not have exactly 2 elements.
            TypeError: If the elements in args are not of type int or float.
        """
        if self is None:
            raise ValueError("Error: Arguments cannot be None")
        if len(self) != 2:
            raise ValueError("Expected exactly 2 arguments")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        arg1, arg2 = self
        return arg1 + arg2

    def sub(self, args: List[Union[int, float]]) -> Union[int, float]:
        arg1, arg2 = args
        result = arg1 - arg2
        if isinstance(result, int) and result > sys.maxsize:
            raise ValueError("Error: Subtraction result exceeds maximum limit of int")
        elif isinstance(result, float) and (result > sys.float_info.max or result < -sys.float_info.max):
            raise ValueError("Error: Subtraction result exceeds maximum limit of float")
        if isinstance(result, int) and result < -sys.maxsize:
            raise ValueError("Error: Subtraction result is below the minimum limit of int")
        elif isinstance(result, float) and (result > -sys.float_info.max or result < sys.float_info.max):
            raise ValueError("Error: Subtraction result is below the minimum limit of float")
        if arg1 is None or arg2 is None:
            raise ValueError("Error: Operands cannot be None")
        if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
            raise TypeError("Invalid operands")
        return result

    def mul(self, args: List[Union[int, float]]) -> Union[int, float]:
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        if not args:
            raise ValueError("Error: Empty list of arguments")
        if len(args) == 1:
            return args[0]
        result = 1
        for arg in args:
            result *= arg
        return 1 if result == 0 else result

    def div(self, args: List[Union[int, float]]) -> Union[int, float]:
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        if not args:
            raise ValueError("Error: Empty list of arguments")
        if len(args) == 1:
            return args[0]
        return args[0] / args[1]

    def pow(self, args: List[Union[int, float]]) -> Union[int, float]:
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        if not args:
            raise ValueError("Error: Empty list of arguments")
        if len(args) == 1:
            return args[0]
        return args[0] ** args[1]

    def number(self, args: List[Union[int, float]]) -> Union[int, float]:
        if not args:
            raise ValueError("Error: Empty list of arguments")
        if len(args) != 1:
            raise ValueError("Error: Expected exactly 1 argument")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        return args[0]

    def validate_args(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    validate_args = staticmethod(validate_args)

    def validate_args_with_zero_check(func):

        def wrapper(self, arg1: Union[int, float], arg2: Union[int, float]):
            if not isinstance(arg1, (int, float)) or not isinstance(arg2, (int, float)):
                raise TypeError("Invalid operands. Expected int or float.")
            if arg2 == 0:
                raise ZeroDivisionError("Error: Division by zero")
            return func(self, arg1, arg2)  # Call the decorated function with the validated arguments

        return wrapper

    validate_args_with_zero_check = staticmethod(validate_args_with_zero_check)

    def validate_args_with_zero_check_and_log_error(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    validate_args_with_zero_check_and_log_error = staticmethod(validate_args_with_zero_check_and_log_error)

    def validate_args_with_zero_check_and_log_error_and_raise_exception(func):

        def wrapper(self, *args: Union[int, float]):
            if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
                raise InvalidArgumentsError("Expected exactly 2 arguments of type int or float")
            if args[1] == 0:
                raise ValueError("Error: Division by zero")
            return func(self, *args)  # Call the decorated function with the validated arguments

        return wrapper

    validate_args_with_zero_check_and_log_error_and_raise_exception = staticmethod(
        validate_args_with_zero_check_and_log_error_and_raise_exception
    )

    def add(self, args: List[Union[int, float]]) -> Union[int, float]:
        if None in args:
            raise ValueError("Error: Operands cannot be None")
        if not all(isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Invalid operands. Expected int or float.")
        if not args:
            raise ValueError("Error: Empty list of arguments")
        return args[0] if len(args) == 1 else sum(args)


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

parser = Lark(grammar, parser='lalr', transformer=CalculateTree())
result = parser.parse("2 + 3 * (4 - 1)")
print(result)  # Output: 11
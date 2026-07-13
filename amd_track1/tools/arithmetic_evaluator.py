"""
Tool 1: Arithmetic Evaluator

Safe expression evaluation for integer and decimal arithmetic,
percentages, ratios, and operator precedence without using eval.
Implements structured CalculationResult and safe function support.
"""

import re
import math
from typing import Optional, Union, List, Any
from dataclasses import dataclass, field
from enum import Enum


class ErrorCode(Enum):
    """Error codes for calculation failures."""
    SUCCESS = "success"
    INVALID_EXPRESSION = "invalid_expression"
    UNSUPPORTED_TOKEN = "unsupported_token"
    DIVISION_BY_ZERO = "division_by_zero"
    EXPRESSION_TOO_LONG = "expression_too_long"
    NUMBER_TOO_LARGE = "number_too_large"
    COMPLEXITY_EXCEEDED = "complexity_exceeded"
    UNSUPPORTED_FUNCTION = "unsupported_function"
    SYNTAX_ERROR = "syntax_error"
    MISMATCHED_PARENTHESES = "mismatched_parentheses"


@dataclass(frozen=True)
class CalculationResult:
    """Structured result of a calculation attempt."""
    success: bool
    value: Optional[Union[int, float]] = None
    normalized_expression: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# Safe functions that can be used in expressions
SAFE_FUNCTIONS = {
    'sqrt': lambda x: math.sqrt(x),
    'round': lambda x, ndigits=None: round(x, ndigits) if ndigits is not None else round(x),
    'abs': lambda x: abs(x),
    'min': lambda *args: min(args),
    'max': lambda *args: max(args),
}

# Maximum expression length to prevent DoS
MAX_EXPRESSION_LENGTH = 1000

# Maximum magnitude for numbers
MAX_NUMBER_MAGNITUDE = 1e100

# Maximum number of tokens in expression
MAX_TOKENS = 200

# Maximum depth of nested operations
MAX_DEPTH = 50


class ArithmeticEvaluator:
    """Safe arithmetic expression evaluator."""
    
    # Supported operators and their precedence
    OPERATORS = {
        '+': (1, lambda a, b: a + b),
        '-': (1, lambda a, b: a - b),
        '*': (2, lambda a, b: a * b),
        '/': (2, lambda a, b: a / b),
        '//': (2, lambda a, b: a // b),
        '%': (2, lambda a, b: a % b),
        '^': (3, lambda a, b: a ** b),
        '**': (3, lambda a, b: a ** b),
    }
    
    # Supported safe functions with precedence and minimum arity
    # min and max can take variable number of arguments
    FUNCTIONS = {
        'sqrt': (4, 1),   # precedence, min_arity
        'abs': (4, 1),
        'round': (4, 2),  # round(number, ndigits)
        'min': (4, 2),
        'max': (4, 2),
    }
    
    # Regex for parsing numbers (int, float, with/without signs)
    # Changed to handle negative numbers at the start or after operators
    NUMBER_PATTERN = r'-?\d+\.?\d*'
    
    # Regex for function names
    FUNCTION_PATTERN = r'[a-zA-Z_][a-zA-Z0-9_]*'
    
    def __init__(self):
        self._cache = {}

    def calculate(self, expression: str) -> CalculationResult:
        """
        Safely evaluate a mathematical expression and return a structured result.
        
        This is the primary interface for the calculator that returns a structured
        CalculationResult instead of raising exceptions.
        
        Args:
            expression: The arithmetic expression to evaluate
            
        Returns:
            CalculationResult with success status, value, or error information
        """
        # Check for empty or whitespace-only expression
        if not expression or expression.strip() == "":
            return CalculationResult(
                success=False,
                normalized_expression=None,
                error_code=ErrorCode.INVALID_EXPRESSION.value,
                error_message="Empty expression"
            )
        
        # Check expression length
        if len(expression) > MAX_EXPRESSION_LENGTH:
            return CalculationResult(
                success=False,
                normalized_expression=expression[:50] + "...",
                error_code=ErrorCode.EXPRESSION_TOO_LONG.value,
                error_message=f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH}"
            )
        
        # Explicit safety checks for dangerous constructs
        dangerous_patterns = [
            r'__import__',
            r'import\s+',
            r'from\s+',
            r'open\s*\(',
            r'os\.',
            r'sys\.',
            r'subprocess\.',
            r'exec\s*\(',
            r'eval\s*\(',
            r'lambda\s*:',
            r'def\s+',
            r'class\s+',
            r'\bfor\s+',
            r'\bwhile\s+',
            r'\bif\s+',
            r'\bwith\s+',
            r'\btry\s+',
            r'\bextend\s*\(',
            r'\b__[a-zA-Z_]+__',  # dunder methods
            r'\.\w+\s*\(',  # method calls
            r'=\s*',  # assignment
            r':\s*=',  # walrus operator
            r'\+\+',  # increment (not standard in Python but catch just in case)
            r'--',  # decrement
            r'\$',  # shell variables
            r'`',  # shell commands
            r'\\',  # escape sequences that might be used for injection
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression):
                return CalculationResult(
                    success=False,
                    normalized_expression=expression[:50] + "...",
                    error_code=ErrorCode.UNSUPPORTED_TOKEN.value,
                    error_message=f"Expression contains unsafe construct matching: {pattern}"
                )
        
        try:
            # Validate and normalize the expression
            normalized = self._normalize_expression(expression)
            
            # Tokenize
            tokens = self.tokenize(normalized)
            
            # Check token count
            if len(tokens) > MAX_TOKENS:
                return CalculationResult(
                    success=False,
                    normalized_expression=normalized,
                    error_code=ErrorCode.COMPLEXITY_EXCEEDED.value,
                    error_message=f"Expression has {len(tokens)} tokens, exceeds maximum of {MAX_TOKENS}"
                )
            
            # Convert to postfix
            postfix = self._shunting_yard(tokens)
            
            # Evaluate postfix
            result = self._evaluate_postfix(postfix, expression)
            
            if isinstance(result, float):
                # Check for infinity or NaN
                if math.isinf(result) or math.isnan(result):
                    return CalculationResult(
                        success=False,
                        normalized_expression=normalized,
                        error_code=ErrorCode.NUMBER_TOO_LARGE.value,
                        error_message="Result is infinite or not a number"
                    )
                
                # Check magnitude
                if abs(result) > MAX_NUMBER_MAGNITUDE:
                    return CalculationResult(
                        success=False,
                        normalized_expression=normalized,
                        error_code=ErrorCode.NUMBER_TOO_LARGE.value,
                        error_message=f"Result magnitude {abs(result)} exceeds maximum of {MAX_NUMBER_MAGNITUDE}"
                    )
            
            return CalculationResult(
                success=True,
                value=result,
                normalized_expression=normalized,
                error_code=None,
                error_message=None
            )
            
        except ValueError as e:
            return CalculationResult(
                success=False,
                normalized_expression=expression[:50] + ("..." if len(expression) > 50 else ""),
                error_code=ErrorCode.INVALID_EXPRESSION.value,
                error_message=str(e)
            )
        except ZeroDivisionError:
            return CalculationResult(
                success=False,
                normalized_expression=expression[:50] + ("..." if len(expression) > 50 else ""),
                error_code=ErrorCode.DIVISION_BY_ZERO.value,
                error_message="Division by zero"
            )
        except Exception as e:
            return CalculationResult(
                success=False,
                normalized_expression=expression[:50] + ("..." if len(expression) > 50 else ""),
                error_code=ErrorCode.SYNTAX_ERROR.value,
                error_message=f"Evaluation error: {str(e)}"
            )

    def _normalize_expression(self, expression: str) -> str:
        """
        Normalize expression by removing whitespace and handling special cases.
        """
        # Handle percentage by converting "X%" to "X * 0.01" but only when % is a percentage sign
        # We need to be careful not to break the modulo operator (10 % 3)
        # Strategy: Use regex to find number% patterns and replace them before removing spaces
        expr = re.sub(r'(\d+\.?\d*)%', r'\1 * 0.01', expression)
        
        # Remove all whitespace
        expr = expr.replace(' ', '')
        
        # Normalize exponentiation operators
        expr = expr.replace('^', '**')
        
        return expr

    def _evaluate_postfix(self, postfix: list, original_expr: str) -> Union[int, float]:
        """
        Evaluate postfix notation with improved error handling and function support.
        """
        eval_stack = []
        
        for token in postfix:
            if re.match(self.NUMBER_PATTERN, token):
                num = self._parse_number(token)
                
                # Check magnitude of input numbers
                if abs(num) > MAX_NUMBER_MAGNITUDE:
                    raise ValueError(f"Number {num} exceeds maximum magnitude of {MAX_NUMBER_MAGNITUDE}")
                
                eval_stack.append(num)
            elif token in self.FUNCTIONS:
                # Function call
                func_name = token
                arity = self.FUNCTIONS[func_name][1]  # Number of arguments
                
                if len(eval_stack) < arity:
                    raise ValueError(f"Not enough arguments for function {func_name}")
                
                # Pop arguments in reverse order
                args = []
                for _ in range(arity):
                    args.append(eval_stack.pop())
                args.reverse()  # Put them back in the right order
                
                # Apply the function
                try:
                    if func_name == 'sqrt':
                        result = math.sqrt(args[0])
                    elif func_name == 'round':
                        if len(args) == 1:
                            result = round(args[0])
                        else:
                            # Convert ndigits to int for round function
                            result = round(args[0], int(args[1]))
                    elif func_name == 'abs':
                        result = abs(args[0])
                    elif func_name == 'min':
                        result = min(args)
                    elif func_name == 'max':
                        result = max(args)
                    else:
                        raise ValueError(f"Unknown function: {func_name}")
                    
                    # Check magnitude of result
                    if isinstance(result, (int, float)) and abs(result) > MAX_NUMBER_MAGNITUDE:
                        raise ValueError(f"Function result {result} exceeds maximum magnitude")
                    
                    eval_stack.append(result)
                except ValueError as e:
                    # Handle domain errors (e.g., sqrt of negative)
                    raise ValueError(f"Function {func_name} error: {str(e)}")
            elif token in self.OPERATORS:
                if len(eval_stack) < 2:
                    raise ValueError(f"Not enough operands for operator {token}")
                
                b = eval_stack.pop()
                a = eval_stack.pop()
                
                # Check for division by zero
                if token in ['/', '//', '%'] and b == 0:
                    raise ZeroDivisionError(f"Division by zero in expression: {original_expr}")
                
                result = self._apply_operator(a, b, token)
                
                # Check magnitude of intermediate results
                if isinstance(result, (int, float)) and abs(result) > MAX_NUMBER_MAGNITUDE:
                    raise ValueError(f"Intermediate result {result} exceeds maximum magnitude")
                
                eval_stack.append(result)
            elif token == ',':
                # Comma is handled during parsing, shouldn't appear in postfix
                continue
            else:
                # Unknown token - should have been caught earlier
                raise ValueError(f"Unknown token: {token}")
        
        if len(eval_stack) != 1:
            raise ValueError("Invalid expression - stack has multiple values")
        
        return eval_stack[0]
    
    def tokenize(self, expression: str) -> list:
        """Tokenize an arithmetic expression into numbers, operators, and functions."""
        # Remove all whitespace
        expr = expression.replace(' ', '')
        
        tokens = []
        i = 0
        while i < len(expr):
            # Check for function names
            if i < len(expr) and expr[i].isalpha():
                # Try to match a function name
                match = re.match(self.FUNCTION_PATTERN, expr[i:])
                if match:
                    func_name = match.group()
                    # Check if this is a valid function
                    if func_name in self.FUNCTIONS:
                        tokens.append(func_name)
                        i += len(func_name)
                        # Next character should be '('
                        if i < len(expr) and expr[i] == '(':
                            tokens.append('(')
                            i += 1
                        else:
                            raise ValueError(f"Expected '(' after function {func_name} at position {i}")
                    else:
                        raise ValueError(f"Unknown function: {func_name}")
                else:
                    raise ValueError(f"Invalid identifier at position {i}: {expr[i]}")
            elif expr[i] == '%':
                # For now, treat all % as modulo operator
                # Percentage support is handled in normalization with a more careful regex
                tokens.append('%')
                i += 1
            # Check for multi-character operators first
            elif expr[i:i+2] in self.OPERATORS:
                tokens.append(expr[i:i+2])
                i += 2
            elif expr[i] in self.OPERATORS:
                # Check if this is a unary minus/plus
                if expr[i] in '+-' and (i == 0 or expr[i-1] == '(' or expr[i-1] in '+-*/%^**'):
                    # This is a unary operator, keep it as part of the number
                    # Parse the sign with the following number
                    sign = expr[i]
                    i += 1
                    # Now parse the number
                    match = re.match(r'\d+\.?\d*', expr[i:])
                    if match:
                        num_str = sign + match.group()
                        tokens.append(num_str)
                        i += len(match.group())
                    else:
                        raise ValueError(f"Expected number after unary {sign} at position {i}")
                else:
                    tokens.append(expr[i])
                    i += 1
            elif expr[i] == '(' or expr[i] == ')':
                tokens.append(expr[i])
                i += 1
            elif expr[i] == ',':
                tokens.append(',')
                i += 1
            else:
                # Parse number
                match = re.match(self.NUMBER_PATTERN, expr[i:])
                if match:
                    num_str = match.group()
                    tokens.append(num_str)
                    i += len(match.group())
                else:
                    raise ValueError(f"Invalid character at position {i}: {expr[i]}")
        
        return tokens
    
    def _parse_number(self, token: str) -> float:
        """Safely parse a numeric token."""
        try:
            if '.' in token:
                return float(token)
            return float(token)  # int will be converted to float
        except ValueError:
            raise ValueError(f"Invalid number: {token}")
    
    def _apply_operator(self, a: float, b: float, op: str) -> float:
        """Apply an operator to two operands."""
        if op not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {op}")
        return self.OPERATORS[op][1](a, b)
    
    def _shunting_yard(self, tokens: list) -> list:
        """Convert infix tokens to postfix (RPN) using Shunting-yard algorithm with function support."""
        output = []
        stack = []
        
        for token in tokens:
            if re.match(self.NUMBER_PATTERN, token):
                output.append(token)
            elif token in self.FUNCTIONS:
                # Function token - push to stack
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                # Pop operators until '('
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses")
                stack.pop()  # Remove the '('
                # If there's a function at the top of the stack, pop it
                if stack and stack[-1] in self.FUNCTIONS:
                    output.append(stack.pop())
            elif token == ',':
                # Comma - pop operators until '('
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if not stack:
                    raise ValueError("Mismatched parentheses or comma outside function")
            elif token in self.OPERATORS:
                # Operator
                precedence = self.OPERATORS[token][0]
                while (stack and stack[-1] != '(' and 
                       self.OPERATORS.get(stack[-1], (0,))[0] >= precedence):
                    output.append(stack.pop())
                stack.append(token)
            else:
                raise ValueError(f"Unknown token in shunting yard: {token}")
        
        while stack:
            if stack[-1] == '(':
                raise ValueError("Mismatched parentheses")
            output.append(stack.pop())
        
        return output
    
    def evaluate(self, expression: str) -> float:
        """
        Evaluate a mathematical expression safely.
        
        Supports:
        - Integer and decimal arithmetic
        - Percentages (convert % to * 0.01 internally)
        - Ratios (a/b)
        - Operator precedence
        - Parentheses
        - Safe functions: sqrt, round, abs, min, max
        
        Args:
            expression: The arithmetic expression to evaluate
            
        Returns:
            The result as a float
            
        Raises:
            ValueError: If expression is invalid
        """
        # Normalize the expression (handles % and ^ -> **)
        normalized = self._normalize_expression(expression)
        
        # Tokenize
        tokens = self.tokenize(normalized)
        
        # Convert to postfix
        postfix = self._shunting_yard(tokens)
        
        # Evaluate postfix
        result = self._evaluate_postfix(postfix, expression)
        
        return result
    
    def evaluate_to_string(self, expression: str) -> str:
        """
        Evaluate and return result as string.
        
        Handles integer results by returning int string when appropriate.
        """
        result = self.evaluate(expression)
        
        # Check if result is an integer
        if result == int(result):
            return str(int(result))
        
        # Format to avoid unnecessary trailing zeros
        return f"{result:.10f}".rstrip('0').rstrip('.')
    
    def validate_expression(self, expression: str) -> bool:
        """Validate if an expression is syntactically valid."""
        try:
            self.evaluate(expression)
            return True
        except (ValueError, TypeError):
            return False
    
    def solve_for_variable(self, equation: str, variable: str = 'x') -> Optional[float]:
        """
        Simple linear equation solver: ax + b = c or ax = b.
        
        This is a simplified solver that handles basic linear equations.
        For more complex equations, the model should be used.
        """
        # Remove spaces
        eq = equation.replace(' ', '')
        
        # Split by equals
        if '=' not in eq:
            return None
        
        left, right = eq.split('=', 1)
        
        # Only handle cases where variable appears once
        if left.count(variable) + right.count(variable) != 1:
            return None
        
        # Case: x = value
        if variable in left and variable not in right:
            if left == variable:
                try:
                    return self.evaluate(right)
                except:
                    return None
        
        # Case: value = x
        if variable in right and variable not in left:
            if right == variable:
                try:
                    return self.evaluate(left)
                except:
                    return None
        
        # Case: ax + b = c or ax = c
        if variable in left:
            # Try to parse: coefficient * variable + constant = value
            # This is simplified - for full support use model
            left_parts = left.replace('+', ' +').replace('-', ' -').split()
            coeff = 1.0
            const = 0.0
            has_var = False
            
            for part in left_parts:
                if variable in part:
                    if part == variable:
                        pass  # coefficient is 1
                    elif part.startswith(variable):
                        # Like x5 or x*5
                        return None  # Too complex for simple solver
                    else:
                        # Try to extract coefficient
                        try:
                            coeff = self.evaluate(part.replace(variable, ''))
                            has_var = True
                        except:
                            return None
                else:
                    try:
                        const = self.evaluate(part)
                    except:
                        return None
            
            if has_var:
                try:
                    right_val = self.evaluate(right)
                    result = (right_val - const) / coeff
                    return result
                except:
                    return None
        
        return None


# Singleton instance
arithmetic_evaluator = ArithmeticEvaluator()

import ast
import re
from typing import List, Dict, Tuple, Optional, Any

class CountdownVerifier:
    """
    Deterministic and secure AST-based verifier for the Countdown math game.
    """
    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else None,
        ast.FloorDiv: lambda a, b: a // b if b != 0 else None,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: +a
    }

    # Explicitly include operator classes because ast.walk visits them as child nodes of BinOp/UnaryOp
    ALLOWED_NODE_TYPES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd
    )

    @staticmethod
    def extract_tags(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning and final equation from XML-like tags <think> and <answer>."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        thought = think_match.group(1).strip() if think_match else None

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else None

        # Fallback if the model produced an opening tag without a closing tag
        if not answer:
            answer_partial = re.search(r"<answer>(.*)", text, re.DOTALL | re.IGNORECASE)
            if answer_partial:
                answer = answer_partial.group(1).strip().split("\n")[0].strip()

        return thought, answer

    @classmethod
    def is_safe_ast(cls, tree: ast.AST) -> bool:
        """Verify that the AST contains ONLY allowed arithmetic nodes and operators."""
        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED_NODE_TYPES):
                return False
            if isinstance(node, ast.BinOp) and type(node.op) not in cls.ALLOWED_OPERATORS:
                return False
            if isinstance(node, ast.UnaryOp) and type(node.op) not in cls.ALLOWED_OPERATORS:
                return False
            if isinstance(node, ast.Constant):
                # Exclude strings, None, and booleans (in Python, bool is a subclass of int)
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    return False
        return True

    @classmethod
    def evaluate_ast(cls, node: ast.AST) -> Optional[float]:
        """Recursively evaluate an AST node, allowing only basic arithmetic operations."""
        if isinstance(node, ast.Expression):
            return cls.evaluate_ast(node.body)

        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)

        elif isinstance(node, ast.UnaryOp):
            op_func = cls.ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                return None
            val = cls.evaluate_ast(node.operand)
            return op_func(val) if val is not None else None

        elif isinstance(node, ast.BinOp):
            op_func = cls.ALLOWED_OPERATORS.get(type(node.op))
            if not op_func:
                return None
            left_val = cls.evaluate_ast(node.left)
            right_val = cls.evaluate_ast(node.right)
            if left_val is None or right_val is None:
                return None
            try:
                return op_func(left_val, right_val)
            except (ZeroDivisionError, OverflowError):
                return None

        return None

    @classmethod
    def extract_numbers_from_ast(cls, node: ast.AST) -> List[int]:
        """Extract all numbers used in the parsed AST expression."""
        numbers = []
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                if float(n.value).is_integer():
                    numbers.append(int(n.value))
                else:
                    numbers.append(n.value)
        return numbers

    @classmethod
    def verify(cls, available_numbers: List[int], target: int, completion: str) -> Dict[str, Any]:
        """
        Full verification of the model completion.
        Returns a dictionary containing formatting, validity, and correctness status.
        """
        thought, answer_str = cls.extract_tags(completion)

        result = {
            "has_format": bool(thought is not None and answer_str is not None),
            "thought": thought,
            "answer_str": answer_str,
            "valid_syntax": False,
            "valid_numbers": False,
            "used_numbers": [],
            "computed_value": None,
            "is_correct": False,
            "error_msg": None
        }

        if not answer_str:
            result["error_msg"] = "Tag <answer> not found or empty"
            return result

        # 1. Parse AST and enforce security whitelist
        try:
            clean_expr = answer_str.split("=")[0].strip()
            tree = ast.parse(clean_expr, mode='eval')
            if not cls.is_safe_ast(tree):
                result["error_msg"] = "Expression contains unallowed elements (only +, -, *, /, parentheses, and numbers)"
                return result
            result["valid_syntax"] = True
        except Exception as e:
            result["error_msg"] = f"AST syntax error: {e}"
            return result

        # 2. Verify number usage constraints
        used_numbers = cls.extract_numbers_from_ast(tree)
        result["used_numbers"] = used_numbers

        available_copy = list(available_numbers)
        valid_nums = True
        for num in used_numbers:
            if num in available_copy:
                available_copy.remove(num)
            else:
                valid_nums = False
                break

        result["valid_numbers"] = valid_nums and (len(used_numbers) > 0)
        if not result["valid_numbers"]:
            result["error_msg"] = f"Invalid numbers or numbers used more than once: {used_numbers} (Available: {available_numbers})"
            return result

        # 3. Evaluate mathematical expression
        computed = cls.evaluate_ast(tree)
        result["computed_value"] = computed

        if computed is None:
            result["error_msg"] = "Evaluation failed (e.g., division by zero)"
            return result

        # 4. Check target equality (with float tolerance for exact divisions)
        if abs(computed - target) < 1e-6:
            result["is_correct"] = True
        else:
            result["error_msg"] = f"Computed result {computed} != target {target}"

        return result

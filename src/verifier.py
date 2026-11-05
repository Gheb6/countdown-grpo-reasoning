import ast
import re
from typing import List, Dict, Tuple, Optional, Any

class CountdownVerifier:
    """
    Verificatore deterministico e sicuro per il gioco Countdown basato su AST (Abstract Syntax Tree).
    """
    ALLOWED_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b if b != 0 else None,
        ast.USub: lambda a: -a
    }

    @staticmethod
    def extract_tags(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Estrae thought e answer dai tag XML-like <think> e <answer>."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        thought = think_match.group(1).strip() if think_match else None

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else None

        if not answer:
            answer_partial = re.search(r"<answer>(.*)", text, re.DOTALL | re.IGNORECASE)
            if answer_partial:
                answer = answer_partial.group(1).strip().split("\n")[0].strip()

        return thought, answer

    @classmethod
    def evaluate_ast(cls, node: ast.AST) -> Optional[float]:
        """Valuta ricorsivamente un nodo AST consentendo solo operazioni matematiche di base."""
        if isinstance(node, ast.Expression):
            return cls.evaluate_ast(node.body)

        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return float(node.n)

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
            except ZeroDivisionError:
                return None

        return None

    @classmethod
    def extract_numbers_from_ast(cls, node: ast.AST) -> List[int]:
        """Estrae tutti i numeri utilizzati nell'espressione AST."""
        numbers = []
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                if float(n.value).is_integer():
                    numbers.append(int(n.value))
                else:
                    numbers.append(n.value)
            elif hasattr(ast, 'Num') and isinstance(n, ast.Num):
                if float(n.n).is_integer():
                    numbers.append(int(n.n))
                else:
                    numbers.append(n.n)
        return numbers

    @classmethod
    def verify(cls, available_numbers: List[int], target: int, completion: str) -> Dict[str, Any]:
        """
        Verifica completa del completamento generato dal modello.
        Restituisce un dizionario con lo stato di formattazione, validita e correttezza.
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
            result["error_msg"] = "Tag <answer> non trovato o vuoto"
            return result

        try:
            clean_expr = answer_str.split("=")[0].strip()
            tree = ast.parse(clean_expr, mode='eval')
            result["valid_syntax"] = True
        except Exception as e:
            result["error_msg"] = f"Errore sintassi AST: {e}"
            return result

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
            result["error_msg"] = f"Numeri non validi o usati piu volte: {used_numbers} (Disponibili: {available_numbers})"
            return result

        computed = cls.evaluate_ast(tree)
        result["computed_value"] = computed

        if computed is None:
            result["error_msg"] = "Valutazione fallita (es. divisione per zero)"
            return result

        if abs(computed - target) < 1e-6:
            result["is_correct"] = True
        else:
            result["error_msg"] = f"Risultato {computed} != target {target}"

        return result

import re
import ast
from typing import List

ALLOWED_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b if b != 0 else None,
    ast.FloorDiv: lambda a, b: a // b if b != 0 else None,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a
}

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

def is_safe_math_ast(tree: ast.AST) -> bool:
    """Check that the AST tree only contains allowed arithmetic nodes and operators."""
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            return False
        if isinstance(node, ast.BinOp) and type(node.op) not in ALLOWED_OPERATORS:
            return False
        if isinstance(node, ast.UnaryOp) and type(node.op) not in ALLOWED_OPERATORS:
            return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return False
    return True

def format_reward_func(completions, **kwargs) -> List[float]:
    """Rewards (+0.2) compliance with the <think>...</think><answer>...</answer> XML structure."""
    rewards = []
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        if re.search(pattern, content.strip(), re.DOTALL | re.IGNORECASE):
            rewards.append(0.2)
        elif "<answer>" in content and "</answer>" in content:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards

def equation_validity_reward_func(completions, numbers, **kwargs) -> List[float]:
    """Rewards (+0.1) if the expression is syntactically valid and uses only allowed numbers."""
    rewards = []
    for comp, nums in zip(completions, numbers):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        ans_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not ans_match:
            rewards.append(0.0)
            continue
        expr = ans_match.group(1).split("=")[0].strip()
        try:
            tree = ast.parse(expr, mode='eval')
            if not is_safe_math_ast(tree):
                rewards.append(0.0)
                continue
            used = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                    used.append(int(n.value) if float(n.value).is_integer() else n.value)
            avail = list(nums)
            valid = True
            for u in used:
                if u in avail:
                    avail.remove(u)
                else:
                    valid = False
                    break
            rewards.append(0.1 if (valid and len(used) > 0) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, numbers, target, **kwargs) -> List[float]:
    """Rewards (+1.0) if the mathematical evaluation strictly matches the target value."""
    rewards = []
    for comp, nums, tgt in zip(completions, numbers, target):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        ans_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if not ans_match:
            rewards.append(0.0)
            continue
        expr = ans_match.group(1).split("=")[0].strip()
        try:
            tree = ast.parse(expr, mode='eval')
            if not is_safe_math_ast(tree):
                rewards.append(0.0)
                continue
            used = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                    used.append(int(n.value) if float(n.value).is_integer() else n.value)
            avail = list(nums)
            valid = True
            for u in used:
                if u in avail:
                    avail.remove(u)
                else:
                    valid = False
                    break
            if not valid or len(used) == 0:
                rewards.append(0.0)
                continue

            def ev(node):
                if isinstance(node, ast.Expression):
                    return ev(node.body)
                elif isinstance(node, ast.Constant):
                    return float(node.value)
                elif isinstance(node, ast.UnaryOp):
                    op = ALLOWED_OPERATORS.get(type(node.op))
                    v = ev(node.operand)
                    return op(v) if op and v is not None else None
                elif isinstance(node, ast.BinOp):
                    op = ALLOWED_OPERATORS.get(type(node.op))
                    l = ev(node.left)
                    r = ev(node.right)
                    return op(l, r) if op and l is not None and r is not None else None
                return None

            computed = ev(tree)
            if computed is not None and abs(computed - tgt) < 1e-6:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

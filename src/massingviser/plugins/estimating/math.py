"""Money and quantity arithmetic.

``Money.amount_minor`` is **minor units** (cents, pence) held as an integer. Estimating multiplies
rates by quantities thousands of times and then sums them; doing that in floating point accumulates
drift that shows up as a BOQ total disagreeing with the sum of its own lines by a few pence, which
destroys trust in the whole document for a reason nobody can find.
"""

from __future__ import annotations

import math as _math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ...kernel import KernelError, Result, err, ok
from ...schema import Money


def money(amount: float, currency: str) -> Money:
    """Build a money value, refusing anything that is not a finite number.

    ``round(nan)`` raises but ``int(nan)`` and float arithmetic do not, so without this a single
    bad factor -- a malformed rate in an imported cost library, a division that produced infinity
    -- propagates silently through ``multiply_money`` into a BOQ line total, an estimate subtotal,
    and a cashflow, with every intermediate check passing. A total of ``nan`` on a tender is worse
    than a raised error.
    """
    if not _math.isfinite(amount):
        raise KernelError(
            "COMMAND_FAILED",
            f"Money amount must be finite, got {amount}.",
            {"amount": amount, "currency": currency},
        )
    return Money(_round_half_away(amount), currency)


def _round_half_away(value: float) -> int:
    """Round half away from zero.

    Python's built-in ``round`` uses banker's rounding, which sends 0.5 to 0 and 1.5 to 2. That is
    the right default for statistics and the wrong one for money: an estimator checking a line by
    hand expects 2.5 pence to become 3.
    """
    return int(_math.floor(value + 0.5)) if value >= 0 else int(_math.ceil(value - 0.5))


def from_major(major: float, currency: str, minor_per_major: int = 100) -> Money:
    """Build Money from a major-unit figure, e.g. ``from_major(12.5, "GBP")`` -> 1250."""
    return money(major * minor_per_major, currency)


def to_major(value: Money, minor_per_major: int = 100) -> float:
    return value.amount_minor / minor_per_major


def _currency_mismatch(a: str, b: str) -> KernelError:
    return KernelError("COMMAND_FAILED", f"Cannot combine {a} with {b}.", {"a": a, "b": b})


def add_money(a: Money, b: Money) -> Result[Money, KernelError]:
    if a.currency != b.currency:
        return err(_currency_mismatch(a.currency, b.currency))
    return ok(money(a.amount_minor + b.amount_minor, a.currency))


def subtract_money(a: Money, b: Money) -> Result[Money, KernelError]:
    if a.currency != b.currency:
        return err(_currency_mismatch(a.currency, b.currency))
    return ok(money(a.amount_minor - b.amount_minor, a.currency))


def multiply_money(value: Money, factor: float) -> Money:
    """Multiply by a quantity, rounding half-away-from-zero.

    Rounds once, at the end. Rounding each component of a composite rate before summing is how a
    unit rate ends up a penny out per line and a project's worth of lines ends up materially wrong.
    """
    return money(value.amount_minor * factor, value.currency)


def sum_money(values: Sequence[Money], currency: str) -> Result[Money, KernelError]:
    total = 0
    for value in values:
        if value.currency != currency:
            return err(_currency_mismatch(currency, value.currency))
        total += value.amount_minor
    return ok(Money(total, currency))


def percent_of(value: Money, percent: float) -> Money:
    return multiply_money(value, percent / 100.0)


def is_zero(value: Money) -> bool:
    return value.amount_minor == 0


# ---------------------------------------------------------------------------------------------
# Quantity expressions
# ---------------------------------------------------------------------------------------------

#: Functions a takeoff rule may call. Anything not listed is refused by name.
SAFE_FUNCTIONS: Mapping[str, Callable[..., float]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": lambda value, digits=0: float(round(value, int(digits))),
    "floor": lambda value: float(_math.floor(value)),
    "ceil": lambda value: float(_math.ceil(value)),
    "sqrt": _math.sqrt,
}

_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3}
_RIGHT_ASSOCIATIVE = {"^", "u-", "u+"}
_UNARY_MINUS = "u-"
_UNARY_PLUS = "u+"

_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<op>\*\*|[-+*/%^(),]))"
)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: Any


def tokenize(expression: str) -> Result[list[_Token], KernelError]:
    tokens: list[_Token] = []
    position = 0
    length = len(expression)
    while position < length:
        if expression[position].isspace():
            position += 1
            continue
        match = _TOKEN.match(expression, position)
        if match is None:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f"Unexpected character {expression[position]!r} at {position} "
                    f"in expression {expression!r}.",
                    {"expression": expression, "position": position},
                )
            )
        position = match.end()
        if match.group("number") is not None:
            tokens.append(_Token("number", float(match.group("number"))))
        elif match.group("name") is not None:
            tokens.append(_Token("name", match.group("name")))
        else:
            operator = match.group("op")
            tokens.append(_Token("op", "^" if operator == "**" else operator))
    return ok(tokens)


def _to_rpn(tokens: Sequence[_Token]) -> Result[list[_Token], KernelError]:
    """Shunting-yard.

    Parsing rather than evaluating is the whole point. A takeoff rule's expression comes out of a
    cost library or a saved project -- it is *data*, and handing data to ``eval`` gives whoever
    wrote that file the ability to run code. The grammar here has no attribute access, no
    subscripting, no imports and no calls except the names in ``SAFE_FUNCTIONS``.
    """
    output: list[_Token] = []
    stack: list[_Token] = []
    arg_counts: list[int] = []
    previous: _Token | None = None

    for index, token in enumerate(tokens):
        if token.kind == "number":
            output.append(token)
        elif token.kind == "name":
            # A name followed by `(` is a call; anything else is a variable.
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and following.kind == "op" and following.value == "(":
                if token.value not in SAFE_FUNCTIONS:
                    return err(
                        KernelError(
                            "COMMAND_FAILED",
                            f'Unknown function "{token.value}". Allowed: '
                            f"{', '.join(sorted(SAFE_FUNCTIONS))}.",
                            {"function": token.value},
                        )
                    )
                stack.append(_Token("func", token.value))
                arg_counts.append(1)
            else:
                output.append(token)
        elif token.value == ",":
            while stack and stack[-1].value != "(":
                output.append(stack.pop())
            if not stack or not arg_counts:
                return err(KernelError("COMMAND_FAILED", "Misplaced comma in expression.", {}))
            arg_counts[-1] += 1
        elif token.value == "(":
            stack.append(token)
        elif token.value == ")":
            while stack and stack[-1].value != "(":
                output.append(stack.pop())
            if not stack:
                return err(KernelError("COMMAND_FAILED", "Unbalanced ')' in expression.", {}))
            stack.pop()
            if stack and stack[-1].kind == "func":
                function = stack.pop()
                output.append(_Token("call", (function.value, arg_counts.pop())))
        else:
            operator = token.value
            # Unary sign: at the start, or straight after another operator or an opening paren.
            if operator in ("-", "+") and (
                previous is None
                or (previous.kind == "op" and previous.value not in (")",))
            ):
                operator = _UNARY_MINUS if operator == "-" else _UNARY_PLUS
            precedence = 4 if operator in (_UNARY_MINUS, _UNARY_PLUS) else _PRECEDENCE.get(operator)
            if precedence is None:
                return err(
                    KernelError(
                        "COMMAND_FAILED", f"Unknown operator {operator!r}.", {"operator": operator}
                    )
                )
            while stack and stack[-1].value != "(":
                top = stack[-1].value
                top_precedence = (
                    4 if top in (_UNARY_MINUS, _UNARY_PLUS) else _PRECEDENCE.get(top, 0)
                )
                if top_precedence > precedence or (
                    top_precedence == precedence and operator not in _RIGHT_ASSOCIATIVE
                ):
                    output.append(stack.pop())
                else:
                    break
            stack.append(_Token("op", operator))
        previous = token

    while stack:
        top = stack.pop()
        if top.value == "(":
            return err(KernelError("COMMAND_FAILED", "Unbalanced '(' in expression.", {}))
        output.append(top)
    return ok(output)


def evaluate_expression(
    expression: str, variables: Mapping[str, float] | None = None
) -> Result[float, KernelError]:
    """Evaluate a takeoff expression such as ``Width * Height`` against element properties.

    Never uses ``eval``. Unknown names fail loudly rather than defaulting to zero -- a rule that
    silently measures nothing produces a bill that looks complete and is not, which is the single
    most expensive failure mode in estimating.
    """
    values = variables or {}

    tokenized = tokenize(expression)
    if not tokenized.ok:
        return err(tokenized.error)
    rpn = _to_rpn(tokenized.value)
    if not rpn.ok:
        return err(rpn.error)

    stack: list[float] = []
    for token in rpn.value:
        if token.kind == "number":
            stack.append(token.value)
            continue
        if token.kind == "call":
            name, argc = token.value
            if len(stack) < argc:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'Not enough arguments for "{name}" in "{expression}".',
                        {"expression": expression},
                    )
                )
            arguments = [stack.pop() for _ in range(argc)][::-1]
            try:
                stack.append(float(SAFE_FUNCTIONS[name](*arguments)))
            except Exception as thrown:  # noqa: BLE001
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'Call to "{name}" failed: {thrown}',
                        {"expression": expression},
                    )
                )
            continue
        if token.kind == "name":
            name = token.value
            if name in values:
                stack.append(float(values[name]))
                continue
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Unknown property "{name}" in expression "{expression}".',
                    {"expression": expression, "name": name, "available": sorted(values)},
                )
            )

        operator = token.value
        if operator in (_UNARY_MINUS, _UNARY_PLUS):
            if not stack:
                return err(KernelError("COMMAND_FAILED", "Malformed expression.", {}))
            operand = stack.pop()
            stack.append(-operand if operator == _UNARY_MINUS else operand)
            continue

        if len(stack) < 2:
            return err(
                KernelError(
                    "COMMAND_FAILED",
                    f'Malformed expression "{expression}".',
                    {"expression": expression},
                )
            )
        right = stack.pop()
        left = stack.pop()
        if operator == "+":
            stack.append(left + right)
        elif operator == "-":
            stack.append(left - right)
        elif operator == "*":
            stack.append(left * right)
        elif operator in ("/", "%"):
            if right == 0:
                return err(
                    KernelError(
                        "COMMAND_FAILED",
                        f'Division by zero in expression "{expression}".',
                        {"expression": expression},
                    )
                )
            stack.append(left / right if operator == "/" else _math.fmod(left, right))
        elif operator == "^":
            try:
                stack.append(float(left**right))
            except (OverflowError, ValueError) as thrown:
                return err(
                    KernelError("COMMAND_FAILED", f"Expression overflowed: {thrown}", {})
                )
        else:
            return err(
                KernelError("COMMAND_FAILED", f"Unknown operator {operator!r}.", {})
            )

    if len(stack) != 1:
        return err(
            KernelError(
                "COMMAND_FAILED",
                f'Malformed expression "{expression}".',
                {"expression": expression},
            )
        )
    result = stack[0]
    if not _math.isfinite(result):
        return err(
            KernelError(
                "COMMAND_FAILED",
                f'Expression "{expression}" produced a non-finite value.',
                {"expression": expression},
            )
        )
    return ok(result)

from enum import Enum
from typing import List, Tuple, Optional

class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


class OpType(Enum):
    NOP  = 0
    ROT  = 1
    ENC  = 2
    CNOT = 3
    ADDQ = 4
    REMQ = 5

Action = Tuple[str, int, int, Optional[int]]

def build_action_list_superset(max_qubits: int, feature_bank_size: int, allow_nop: bool = True) -> List[Action]:
    actions: List[Action] = []
    actions.append(("ADD_QUBIT", 0, 0, 0))
    actions.append(("REMOVE_QUBIT", 0, 0, 0))

    for q in range(max_qubits):
        for ax in [Axis.X, Axis.Y, Axis.Z]:
            actions.append(("ROT", ax.value, q, None))

    # for q in range(max_qubits):
    #     for ax in [Axis.X, Axis.Y, Axis.Z]:
    #         for b in range(feature_bank_size):
    #             actions.append(("ENC", ax.value, q, b))
    # ENC actions: feature index is semantic (pixel OR patch)
    for q in range(max_qubits):
        for ax in [Axis.X, Axis.Y, Axis.Z]:
            for fid in range(feature_bank_size):
                actions.append(("ENC", ax.value, q, fid))
    for c in range(max_qubits):
        for t in range(max_qubits):
            if c != t:
                actions.append(("CNOT", c, t, 0))

    if allow_nop:
        actions.append(("NOP", 0, 0, 0))

    return actions


def action_is_valid_for_qubits(action: Action, current_n_qubits: int, current_bank_k: int) -> bool:
    kind = action[0]
    if kind in ("ADD_QUBIT", "REMOVE_QUBIT", "NOP"):
        return True
    if kind == "ROT":
        _, ax, q, _ = action
        return (0 <= q < current_n_qubits)
    if kind == "ENC":
        _, ax, q, b = action
        return (0 <= q < current_n_qubits) and (current_bank_k > 0) and (0 <= b < current_bank_k)
    if kind == "CNOT":
        _, c, t, _ = action
        return (0 <= c < current_n_qubits) and (0 <= t < current_n_qubits) and (c != t)
    return False
"""Conservative local Season-2 combat estimates; never a claim of server authority.

This model deliberately separates *attack generated* from *incoming cancelled*.
The early-game rule doubles cancellation effectiveness, NOT outgoing damage.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Exchange:
    generated: int
    cancelled: int
    remaining: int
    sent: int
    early_double: bool


def exchange(generated, incoming, pieces_placed=0, sent_upper_bound=0,
             meter_confident=False, early_rule=True):
    generated = max(0, int(generated))
    incoming = max(0, int(incoming))
    # Conservative eligibility: upper bound on previously sent attacks means
    # we only claim the 2x cancellation if the condition is definitely plausible.
    early = bool(early_rule and meter_confident and pieces_placed < 14
                 and incoming > max(0, int(sent_upper_bound)))
    cancelled = min(incoming, generated * (2 if early else 1))
    used = (cancelled + 1) // 2 if early else cancelled
    return Exchange(generated, cancelled, incoming - cancelled,
                    max(0, generated - used), early)

"""
select suitable border line from candidates
"""

import logging
from dataclasses import dataclass

from experiments.CharSeg.protetypes.context import Context, Borderline


@dataclass
class DPselecterConfig:
    idonknow: int = 10  # some config values will be here


class DPselecter:
    def __init__(self) -> None:
        self.config = DPselecterConfig()

    def process(self, context: Context):
        logging.info("分割境界線の候補から、採用する境界線をDPで選択します。")

        candidates = context.candidates

        if candidates is None:
            raise ValueError("contextのcandidatesがNoneです。")

        context.selected = self.select_borderline(
            candidates,
            costs=context.candidate_costs,
        )

    def select_borderline(
        self,
        candidates: list[Borderline],
        *,
        costs: list[float] | None = None,
    ) -> list[Borderline]:
        if not candidates:
            return []

        # TODO: DP による選択。暫定で最小コスト候補を1本採用。
        if costs is not None and len(costs) == len(candidates):
            best_idx = min(range(len(costs)), key=costs.__getitem__)
            return [candidates[best_idx]]

        return [candidates[0]]

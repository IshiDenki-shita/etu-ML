"""
select suitable border line from candidates
"""

from dataclasses import dataclass

from experiments.CharSeg.protetypes.context import Context


@dataclass
class DPselecterConfig:
    idonknow: int = 10  # some config values will be here


class DPselecter:
    def __init__(self) -> None:
        self.config = DPselecter()

    def process(self, context: Context):
        ...
        # the DP process will be here

from __future__ import annotations
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Type, TypeVar, Union

from sprouthdl.aggregate.hdl_aggregate import HDLAggregate
from sprouthdl.sprouthdl import Expr, Signal, Wire
from sprouthdl.sprouthdl_module import iter_values

T_Record = TypeVar("T_Record", bound="AggregateRecordDynamic")


class AggregateRecordDynamic(HDLAggregate):

    def to_list_first_level(self) -> List[Expr | "HDLAggregate"]:

        list_first_level: List[Expr | "HDLAggregate"] = []
        for v in iter_values(self, allow_to_list=False):
            if isinstance(v, Expr) or isinstance(v, HDLAggregate):
                list_first_level.append(v)
        return list_first_level
"""
HierarchicalAggregator

Example of a hierarchical aggregation orchestrator.
Top-level aggregator calls regional aggregators asynchronously and then combines.

This is a template; replace regional aggregator implementation with your real RPC/edge aggregator.
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Callable, Awaitable
import asyncio
from .. import logger

class RegionalAggregator:
    """
    Placeholder regional aggregator. Replace with real implementation that aggregates a set of client updates.
    """

    def __init__(self, region_name: str, agg_fn: Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]]):
        self.region_name = region_name
        self.agg_fn = agg_fn

    async def aggregate(self, updates: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        # in real world this may be an RPC to a regional server
        await asyncio.sleep(0)  # yield control
        return self.agg_fn(list(updates))


class HierarchicalAggregator:
    def __init__(self, cluster_strategy: str = "geographic"):
        self.cluster_strategy = cluster_strategy
        self._regional_aggregators: Dict[str, RegionalAggregator] = {}
        self.global_aggregator: RegionalAggregator | None = None

    def register_regional_aggregator(self, region: str, aggregator: RegionalAggregator):
        self._regional_aggregators[region] = aggregator

    def set_global_aggregator(self, aggregator: RegionalAggregator):
        self.global_aggregator = aggregator

    def _group_by_region(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Group client updates by region.
        For demo, we expect client_updates: client_id -> {"region": region, "update": {param: arr}}
        """
        regions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for cid, info in client_updates.items():
            region = info.get("region", "default")
            upd = info.get("update", {})
            regions.setdefault(region, {})[cid] = upd
        return regions

    async def hierarchical_aggregate(self, client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        grouped = self._group_by_region(client_updates)
        # 1) region-level aggregation
        region_tasks = {}
        for region, updates in grouped.items():
            agg = self._regional_aggregators.get(region)
            if agg is None:
                # fallback: use a trivial aggregator that averages numeric arrays elementwise
                def avg_agg(arrs):
                    import numpy as np
                    if not arrs:
                        return {}
                    keys = set(k for u in arrs for k in u.keys())
                    out = {}
                    for k in keys:
                        vals = [u[k] for u in arrs if k in u]
                        out[k] = (sum(map(lambda x: x.astype(float), vals)) / len(vals))
                    return out
                agg = RegionalAggregator(region, avg_agg)
            region_tasks[region] = asyncio.create_task(agg.aggregate(list(updates.values())))
        region_results = {}
        for region, task in region_tasks.items():
            try:
                region_results[region] = await task
            except Exception as e:
                logger.secure_log("warning", "regional aggregation failed", region=region, err=str(e))
                region_results[region] = {}
        # 2) global aggregation
        if self.global_aggregator:
            global_res = await self.global_aggregator.aggregate(region_results.values())
            return global_res
        # fallback: merge region_results by summation where possible
        merged = {}
        for rres in region_results.values():
            for k, v in rres.items():
                merged[k] = merged.get(k, 0) + v
        return merged


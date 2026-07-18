"""Indicator discovery tools."""
import inspect

from ... import indicators as ta


def register_indicator_tools(mcp):
    @mcp.tool()
    def list_indicators() -> dict:
        """List all available technical indicators (call get_indicator_details for signatures)."""
        names = sorted(ta.__all__)
        return {"count": len(names), "indicators": names,
                "note": "Use like: ta.sma(self.candles, 20) or ta.sma(self.candles, 20, sequential=True)."}

    @mcp.tool()
    def get_indicator_details(indicator_name: str) -> dict:
        """Return the signature, parameters, and docstring for one indicator."""
        fn = getattr(ta, indicator_name, None)
        if fn is None or not callable(fn):
            return {"error": "not_found", "message": f'Indicator "{indicator_name}" not found.',
                    "available": sorted(ta.__all__)}
        sig = inspect.signature(fn)
        params = []
        for p in sig.parameters.values():
            params.append({"name": p.name,
                           "default": None if p.default is inspect._empty else repr(p.default)})
        return {"name": indicator_name, "signature": f"{indicator_name}{sig}",
                "parameters": params, "doc": (fn.__doc__ or "").strip(),
                "usage": f"ta.{indicator_name}(self.candles)  # latest value; add sequential=True for the full series"}

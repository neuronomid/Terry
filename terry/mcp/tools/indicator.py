"""Indicator discovery tools."""
import inspect
import importlib

from ... import indicators as ta


def register_indicator_tools(mcp):
    @mcp.tool()
    def list_indicators() -> dict:
        """List all available technical indicators (call get_indicator_details for signatures)."""
        names = sorted(ta.__all__)
        return {"status": "success", "count": len(names), "indicators": names,
                "message": f"Found {len(names)} indicators available in Terry",
                "note": "Use like: ta.sma(self.candles, 20) or ta.sma(self.candles, 20, sequential=True)."}

    @mcp.tool()
    def get_indicator_details(indicator_name: str) -> dict:
        """Return the signature, parameters, and docstring for one indicator."""
        fn = getattr(ta, indicator_name, None)
        if fn is None or not callable(fn):
            return {"status": "error", "error": f"Indicator '{indicator_name}' not found",
                    "error_code": "not_found", "indicator_name": indicator_name,
                    "message": f'Indicator "{indicator_name}" not found.',
                    "available": sorted(ta.__all__)}
        sig = inspect.signature(fn)
        params = {}
        parameter_list = []
        for p in sig.parameters.values():
            info = {"name": p.name,
                    "default": None if p.default is inspect._empty else p.default,
                    "annotation": None if p.annotation is inspect._empty else str(p.annotation),
                    "kind": str(p.kind)}
            params[p.name] = info
            parameter_list.append(info)
        namedtuple_info = None
        try:
            module = importlib.import_module(fn.__module__)
            for name, value in vars(module).items():
                if hasattr(value, "_fields") and hasattr(value, "_field_defaults"):
                    namedtuple_info = {"name": name, "fields": list(value._fields),
                                       "defaults": dict(value._field_defaults)}
                    break
        except (ImportError, AttributeError):
            pass
        usage = (f"ta.{indicator_name}(self.candles)  # latest value; "
                 "add sequential=True for the full series")
        doc = (fn.__doc__ or "").strip()
        return {"status": "success", "name": indicator_name,
                "indicator_name": indicator_name, "signature": str(sig),
                "display_signature": f"{indicator_name}{sig}",
                "parameters": params, "parameter_list": parameter_list,
                "return_annotation": (None if sig.return_annotation is inspect._empty
                                      else str(sig.return_annotation)),
                "doc": doc, "docstring": doc, "namedtuple_info": namedtuple_info,
                "usage": usage, "usage_example": usage,
                "file_path": inspect.getsourcefile(fn),
                "message": f"Successfully retrieved details for indicator: {indicator_name}"}

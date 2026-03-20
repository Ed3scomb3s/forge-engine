import forge_engine as fe


class BuyAndHoldLongStrategy(fe.Strategy):
    """Open one long position and hold it until the session ends."""

    def __init__(self, margin_pct: float = 1.0):
        super().__init__()
        self.margin_pct = float(margin_pct)
        self._opened = False

    def on_candle(self, candle, state, events):
        if self._opened:
            return
        if state.get("position") or state.get("open_order"):
            return
        price = float(candle.get("close", 0.0) or 0.0)
        if price <= 0:
            return
        self.create_order("LONG", price=price, margin_pct=self.margin_pct)
        self._opened = True


class BuyAndHoldShortStrategy(fe.Strategy):
    """Open one short position and hold it until the session ends."""

    def __init__(self, margin_pct: float = 1.0):
        super().__init__()
        self.margin_pct = float(margin_pct)
        self._opened = False

    def on_candle(self, candle, state, events):
        if self._opened:
            return
        if state.get("position") or state.get("open_order"):
            return
        price = float(candle.get("close", 0.0) or 0.0)
        if price <= 0:
            return
        self.create_order("SHORT", price=price, margin_pct=self.margin_pct)
        self._opened = True

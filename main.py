import forge_engine

# Create session
session = forge_engine.create_session(
    symbol="BTCUSDT_PERP",
    start_date="2025-09-01T00:00:00Z",
    end_date="2025-12-31T23:59:00Z",
    starting_cash=10000.0,
    leverage=2.0,
    margin_mode="cross",  # or "isolated"
    warmup_candles=10,
    timeframe="1h",
    slippage_pct=0.001, # A typical value for crypto futures would be 0.0005 to 0.002 (0.05% to 0.2%) depending on market conditions and position size.
)

# Register example indicators (can customize as needed)
try:
    forge_engine.register_indicators(session, [
        forge_engine.SMA(10, source="close", label="SMA10"),
        forge_engine.SMA(20, source="close", label="SMA20"),
        forge_engine.EMA(20, source="close", label="EMA20"),
        forge_engine.RSI(14, source="close", label="RSI14"),
        forge_engine.ATR(14, label="ATR14"),
        forge_engine.BollingerBands(20, 2.0, source="close", label="BB20x2"),
        forge_engine.MACD(12, 26, 9, source="close", label="MACD12269"),
    ])
except Exception:
    pass

# Warmup
warmups = forge_engine.get_warmup_candles(session)
print(f"Warmup candles fetched: {len(warmups)}")

# Example order submitted before first candle (eligible from index 0)
# API uses margin_pct sizing based on current equity (cash + realized_pnl + unrealized_pnl).
# Example below allocates 80% of equity at the stop price.
resp = forge_engine.create_order(session, side="LONG", price=107621, margin_pct=0.1, tp=114000, sl=107000)
print("Create order response:", resp)

input("Press Enter to start stepping through session candles...")


def _min_order_view(o: dict) -> dict:
    return {
        "id": o["id"],
        "type": o["order_type"],
        "side": o["side"],
        "price": o["price"],
        "qty": o["quantity"],
        "eligible": o["eligible_from_index"],
    }

quit_flag = False
for tick in forge_engine.step_session(session):
    candle = tick["candle"]
    state = tick["state"]
    events = tick["events"]

    print(f"\nCandle {state['candle_index']} @ {candle['open_time']}")
    print(f"OHLC: O={candle['open']} H={candle['high']} L={candle['low']} C={candle['close']}")
    print("Market:", {
        "volume": candle.get("volume"),
        "quote_volume": candle.get("quote_asset_volume"),
        "trades": candle.get("number_of_trades"),
        "taker_buy_base": candle.get("taker_buy_base_asset_volume"),
        "taker_buy_quote": candle.get("taker_buy_quote_asset_volume"),
    })

    print("Events:", events)

    inds = state.get("indicators") or {}
    if inds:
        print("Indicators:", inds)

    print("Wallet:", {
        "cash": state["cash"],
        "used_initial_margin": state.get("used_initial_margin"),
        "equity": state["equity"],
        "realized_pnl": state["realized_pnl"],
        "unrealized_pnl": state["unrealized_pnl"],
        "insurance_fund": state["insurance_fund"],
    })

    if state["position"]:
        p = state["position"]
        print("Position:", {
            "side": p["side"],
            "size": p["size"],
            "entry": p["entry_price"],
            "margin": p["margin"],
            "leverage": p["leverage"],
            "liq": p["liquidation_price"],
        })
    if state["open_order"]:
        print("Open order:", _min_order_view(state["open_order"]))
    if state["close_request"]:
        print("Close request:", _min_order_view(state["close_request"]))
    if state["tp"]:
        print("TP:", _min_order_view(state["tp"]))
    if state["sl"]:
        print("SL:", _min_order_view(state["sl"]))

    while True:
        cmd = input("Command [enter=next, o=open, c=close, x=cancel, q=quit]: ").strip().lower()
        if cmd == "q":
            quit_flag = True
            break
        elif cmd == "o":
            side = input("  side [LONG/SHORT]: ").strip().upper()
            price = float(input("  price: ").strip())
            mp = float(input("  margin_pct (fraction of equity, e.g., 0.01 = 1%): ").strip())
            tp_in = input("  tp (blank for none): ").strip()
            tp = float(tp_in) if tp_in else None
            sl_in = input("  sl (blank for none): ").strip()
            sl = float(sl_in) if sl_in else None
            resp = forge_engine.create_order(session, side=side, price=price, margin_pct=mp, tp=tp, sl=sl)
            print("  create_order:", resp)
            if isinstance(resp, dict) and resp.get("status") == "rejected":
                continue
            break
        elif cmd == "c":
            price = float(input("  close price: ").strip())
            print("  close_order:", forge_engine.close_order(session, price=price))
            break
        elif cmd == "x":
            oid = input("  order_id to cancel: ").strip()
            print("  cancel_order:", forge_engine.cancel_order(session, oid))
            break
        else:
            break
    if quit_flag:
        break

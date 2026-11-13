import pandas as pd
from src.features import prepare_features
BASE_TIME = pd.Timestamp("2026-01-01T12:00:00Z")
def create_event(event_id, minutes, token_in, token_out):
    return {
        "event_id": event_id,
        "source": "uniswap",
        "chain": "ethereum",
        "protocol": "uniswap_v3",
        "venue_type": "amm",
        "timestamp": BASE_TIME + pd.Timedelta(minutes=minutes),
        "tx_hash": event_id,
        "evt_index": 0,
        "actor_wallet": "wallet_a",
        "counterparty_wallet": None,
        "pool": "pool_1",
        "token_in": token_in,
        "token_out": token_out,
        "amount_in": 1.0,
        "amount_out": 1.0,
        "amount_usd": None,
        "amounts_normalized": True,
        "role_note": "test data",
    }
def test_reverse_direction_requires_opposite_trade():
    # Create three transactions
    trades = pd.DataFrame([
        create_event("trade_1", 0, "USDC", "WETH"),
        create_event("trade_2", 10, "USDC", "WETH"),
        create_event("trade_3", 20, "WETH", "USDC"),
    ])
# Empty transfer dataset
    transfers = pd.DataFrame()

# Generate features
    features = prepare_features(
    trades,
    transfers
    ).set_index("event_id")

# Verify reverse-direction calculations
    assert features.loc["trade_1", "prior_reverse_1h"] == 0
    assert features.loc["trade_2", "prior_reverse_1h"] == 0
    assert features.loc["trade_3", "prior_reverse_1h"] == 2

# Verify previous liquidity pool history
    assert features.loc["trade_3", "prior_pool_1h"] == 2
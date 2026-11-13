-- ASTRAQUANT PS18 | Uniswap V3, Ethereum | pilot week, UTC
-- Verify this project/version combination returns records before scaling.
-- One row is a swap leg; a transaction may have multiple legs.
SELECT
    blockchain, project, version, block_time, block_number,
    tx_hash, evt_index, tx_from, tx_to, taker, maker,
    project_contract_address,
    token_bought_address, token_sold_address,
    token_bought_symbol, token_sold_symbol,
    token_bought_amount, token_sold_amount, amount_usd
FROM dex.trades
WHERE blockchain = 'ethereum'
  AND project = 'uniswap'
  AND version = '3'
  AND block_time >= TIMESTAMP '2026-10-01 00:00:00'
  AND block_time <  TIMESTAMP '2026-10-08 00:00:00'
ORDER BY block_time, tx_hash, evt_index
LIMIT 100000;

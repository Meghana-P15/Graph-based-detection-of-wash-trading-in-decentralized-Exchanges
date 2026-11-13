-- ASTRAQUANT PS18 | transfer-path evidence for selected Uniswap wallets
-- Pilot query only. CAUTION: transfer tables are large and scans may use credits.
-- 'from' / 'to' can be wallet, pool, router, or other contract.
WITH wallets AS (
  SELECT DISTINCT tx_from AS wallet
  FROM dex.trades
  WHERE blockchain = 'ethereum'
    AND project = 'uniswap'
    AND version = '3'
    AND tx_from IS NOT NULL
    AND block_time >= TIMESTAMP '2026-10-01 00:00:00'
    AND block_time <  TIMESTAMP '2026-10-08 00:00:00'
  LIMIT 200
)
SELECT
  t.blockchain, t.block_time, t.tx_hash, t.evt_index,
  t."from" AS from_address, t."to" AS to_address,
  t.contract_address AS token_address, t.symbol AS token_symbol,
  t.amount, t.amount_usd
FROM tokens.transfers t
WHERE t.blockchain = 'ethereum'
  AND t.block_time >= TIMESTAMP '2026-10-01 00:00:00'
  AND t.block_time <  TIMESTAMP '2026-10-08 00:00:00'
  AND (
    t."from" IN (SELECT wallet FROM wallets)
    OR t."to" IN (SELECT wallet FROM wallets)
  )
ORDER BY t.block_time, t.tx_hash, t.evt_index
LIMIT 100000;

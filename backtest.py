import json # standard library, please let me know if not allowed.
import pandas as pd
import numpy as np


df = pd.read_csv('l1_day.csv')
asks = df[(df['depth'] == 0) & (df['side'] == 'A')]
asks = asks.sort_values(by=["ts_event", "publisher_id", "sequence", "ts_recv"])
asks = asks.drop_duplicates(subset=["ts_event", "publisher_id"], keep='first')

venue_ids = asks['publisher_id'].unique()
fee_map    = { vid: 0.0030 for vid in venue_ids }
rebate_map = { vid: 0.0002 for vid in venue_ids }


def computeCost(split, venues, orderSize, extraSharePenalty, underSharePenalty, queueRiskPenalty):
  executed = 0
  cashSpent = 0
  for i in range(len(venues)):
    exe = min(split[i], venues[i]['ask_size'])
    executed += exe
    cashSpent += exe * (venues[i]['ask'] + venues[i]['fee'])
    makerRebate = max(split[i] - exe, 0) * venues[i]['rebate']
    cashSpent -= makerRebate

  underfill = max(orderSize - executed, 0)
  overfill = max(executed - orderSize, 0)
  riskPenalty = queueRiskPenalty * (underfill + overfill)
  costPenalty = underSharePenalty * underfill + extraSharePenalty * overfill
  return cashSpent + riskPenalty + costPenalty

def allocate(orderSize, venues, extraSharePenalty, underSharePenalty, queueRiskPenalty, step):
  splits = [[]]

  for v in range(len(venues)):
    newSplits = []
    for alloc in splits:
      used = sum(alloc)
      maxV = min(orderSize - used, venues[v]['ask_size'])
      for q in range(0, maxV + 1, step):
        newSplits.append(alloc + [q])
    splits = newSplits

  bestCost = float('inf')
  bestSplit = []
  for alloc in splits:
    if sum(alloc) != orderSize:
      continue
    cost = computeCost(alloc, venues, orderSize, extraSharePenalty, underSharePenalty, queueRiskPenalty)
    if cost < bestCost:
      bestCost = cost
      bestSplit = alloc
  return bestSplit, bestCost

def backtest(venuesDF, orderSize, extraSharePenalty, underSharePenalty, queueRiskPenalty, step=100):
    remainingShares = orderSize
    totalCost       = 0.0
    totalExec       = 0

    for event in sorted(venuesDF['ts_event'].unique()):
        if remainingShares <= 0:
            break

        subsection = venuesDF[venuesDF['ts_event'] == event]

        venues = [
            {
                'ask':      row.ask_px_00,
                'ask_size': int(row.ask_sz_00),
                'fee':      fee_map.get(row.publisher_id, 0.0),
                'rebate':   rebate_map.get(row.publisher_id, 0.0),
            }
            for row in subsection.itertuples()
        ]

        target = remainingShares

        split, cost = allocate(target, venues,
                               extraSharePenalty, underSharePenalty, queueRiskPenalty,
                               step=step)
        if split is None:
            continue

        executed = sum(min(q, v['ask_size']) for q, v in zip(split, venues))
        if executed == 0:
            continue

        totalCost       += cost
        totalExec       += executed
        remainingShares -= executed

    if remainingShares > 0:
        worst = max(v['ask'] + v['fee'] for v in venues)
        totalCost += remainingShares * worst
        totalCost += remainingShares * underSharePenalty
        totalCost += remainingShares * queueRiskPenalty
        totalExec += remainingShares
        remainingShares = 0

    filled = totalExec
    print("Total shares bought:", filled)
    return totalCost, totalExec, totalCost / totalExec

def backtest_best_ask(venuesDF, orderSize):
    remaining = orderSize
    total_cost = 0.0
    total_exec = 0

    for t in sorted(venuesDF['ts_event'].unique()):
        if remaining <= 0:
            break

        snapshot = venuesDF[venuesDF['ts_event'] == t]
        best = min(
            snapshot.itertuples(),
            key=lambda r: (r.ask_px_00 + fee_map.get(r.publisher_id,0.0))
        )
        price, size = best.ask_px_00 + fee_map.get(best.publisher_id,0.0), best.ask_sz_00
        exe = min(size, remaining)
        total_cost += exe * price
        total_exec += exe
        remaining -= exe

    if remaining > 0:
        worst = max(
            (row.ask_px_00 + fee_map.get(row.publisher_id,0.0))
            for row in snapshot.itertuples()
        )
        total_cost += remaining * worst
        total_exec += remaining

    return total_cost, total_exec, total_cost / total_exec

def backtest_twap(venuesDF, orderSize):
    venuesDF = venuesDF.copy()
    
    if isinstance(venuesDF['ts_event'].iloc[0], str):
        venuesDF['datetime'] = pd.to_datetime(venuesDF['ts_event'])
        venuesDF['bucket'] = venuesDF['datetime'].astype('int64') // 1_000_000_000 // 60
    else:
        venuesDF['bucket'] = (venuesDF['ts_event'] // 60_000_000_000).astype(int)
    
    buckets = venuesDF['bucket'].unique()
    N = len(buckets)
    per_bucket = int(np.ceil(orderSize / N))
    
    remaining = orderSize
    total_cost = 0.0
    total_exec = 0
    last_snapshot = None

    for bucket_id, bucket_data in venuesDF.groupby('bucket'):
        if remaining <= 0:
            break
        
        best = min(
            bucket_data.itertuples(),
            key=lambda r: (r.ask_px_00 + fee_map.get(r.publisher_id, 0.0))
        )
        
        price = best.ask_px_00 + fee_map.get(best.publisher_id, 0.0)
        size = best.ask_sz_00
        exe = min(size, per_bucket, remaining)
        total_cost += exe * price
        total_exec += exe
        remaining -= exe
        
        last_snapshot = bucket_data

    if remaining > 0 and last_snapshot is not None:
        worst = max(
            (row.ask_px_00 + fee_map.get(row.publisher_id, 0.0))
            for row in last_snapshot.itertuples()
        )
        total_cost += remaining * worst
        total_exec += remaining

    return total_cost, total_exec, total_cost / total_exec

def backtest_vwap(venuesDF, orderSize):
    remaining = orderSize
    total_cost = 0.0
    total_exec = 0

    for t in sorted(venuesDF['ts_event'].unique()):
        if remaining <= 0:
            break
        snapshot = venuesDF[venuesDF['ts_event'] == t]
        depths = [row.ask_sz_00 for row in snapshot.itertuples()]
        total_depth = sum(depths)
        allocs = [
            int(np.floor(remaining * row.ask_sz_00 / total_depth))
            for row in snapshot.itertuples()
        ]
        for qty, row in zip(allocs, snapshot.itertuples()):
            price = row.ask_px_00 + fee_map.get(row.publisher_id,0.0)
            exe = min(qty, row.ask_sz_00, remaining)
            total_cost += exe * price
            total_exec += exe
            remaining -= exe
        if remaining <= 0:
            break

    if remaining > 0:
        worst = max(row.ask_px_00 + fee_map.get(row.publisher_id,0.0)
                    for row in snapshot.itertuples())
        total_cost += remaining * worst
        total_exec += remaining

    return total_cost, total_exec, total_cost / total_exec


def main():
    lambda_over_vals = [0.0, 0.05, 0.1, 0.2, 0.5]
    lambda_under_vals = [0.0, 0.05, 0.1, 0.2, 0.5]
    theta_vals = [0.0, 0.05, 0.1, 0.2]
    best = (None, float('inf'))
    for lo in lambda_over_vals:
        for lu in lambda_under_vals:
            for th in theta_vals:
                cost, exe, avg = backtest(asks, 5000, lo, lu, th, step=100)
                if avg < best[1]: 
                    best = ((lo, lu, th, cost, exe, avg), avg)
    lo, lu, th, cost, exe, avg = best[0]
    b_cost, b_ex, b_avg = backtest_best_ask(asks, 5000)
    t_cost, t_ex, t_avg = backtest_twap(asks, 5000)
    v_cost, v_ex, v_avg = backtest_vwap(asks, 5000)
    out = {
        "best_params": {"lambda_over": lo, "lambda_under": lu, "theta_queue": th},
        "optimal":    {"cash": cost, "avg_price": avg},
        "best_ask":   {"cash": b_cost, "avg_price": b_avg},
        "twap":       {"cash": t_cost, "avg_price": t_avg},
        "vwap":       {"cash": v_cost, "avg_price": v_avg},
        "savings_bps": {
            "vs_best_ask": 10000*(b_avg-avg)/b_avg,
            "vs_twap":     10000*(t_avg-avg)/t_avg,
            "vs_vwap":     10000*(v_avg-avg)/v_avg
        }
    }
    print(json.dumps(out, indent=2))
    
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results.json")

if __name__ == '__main__':
    main()




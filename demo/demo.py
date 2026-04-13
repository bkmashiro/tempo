#!/usr/bin/env python3
"""
Tempo Demo — Adaptive Rate Limiter with Rhythm Detection

This demo simulates four different client behaviors hitting the same
adaptive rate limiter and shows how each gets classified and treated
differently.
"""

import sys
import os
import random

# Add src to path for demo purposes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tempo.clock import ManualClock
from tempo.detector import DetectorConfig, RequestPattern
from tempo.adaptive import AdaptiveRateLimiter, PolicySet


def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def simulate_bot(clock: ManualClock, limiter: AdaptiveRateLimiter, key: str) -> dict:
    """Simulate a bot: perfectly periodic requests at 500ms intervals."""
    allowed = 0
    denied = 0
    for _ in range(80):
        decision = limiter.check(key)
        if decision.allowed:
            allowed += 1
        else:
            denied += 1
        clock.advance(0.5)
    return {"allowed": allowed, "denied": denied}


def simulate_human(clock: ManualClock, limiter: AdaptiveRateLimiter, key: str) -> dict:
    """Simulate a human: bursts of clicks separated by think-time."""
    rng = random.Random(7812)
    allowed = 0
    denied = 0
    for _ in range(12):
        # Think time
        clock.advance(rng.uniform(2.0, 6.0))
        # Burst of clicks
        for _ in range(rng.randint(3, 7)):
            decision = limiter.check(key)
            if decision.allowed:
                allowed += 1
            else:
                denied += 1
            clock.advance(rng.uniform(0.05, 0.25))
    return {"allowed": allowed, "denied": denied}


def simulate_batch_ramp(clock: ManualClock, limiter: AdaptiveRateLimiter, key: str) -> dict:
    """Simulate a batch job ramping up: intervals decrease over time."""
    allowed = 0
    denied = 0
    for i in range(60):
        decision = limiter.check(key)
        if decision.allowed:
            allowed += 1
        else:
            denied += 1
        interval = max(0.1, 3.0 - i * 0.05)
        clock.advance(interval)
    return {"allowed": allowed, "denied": denied}


def simulate_batch_steady(clock: ManualClock, limiter: AdaptiveRateLimiter, key: str) -> dict:
    """Simulate steady batch processing: very fast, consistent requests."""
    rng = random.Random(4082)
    allowed = 0
    denied = 0
    for _ in range(80):
        decision = limiter.check(key)
        if decision.allowed:
            allowed += 1
        else:
            denied += 1
        clock.advance(0.02 + rng.gauss(0, 0.002))
    return {"allowed": allowed, "denied": denied}


def print_analysis(limiter: AdaptiveRateLimiter, key: str, label: str, stats: dict) -> None:
    analysis = limiter.get_analysis(key)
    pattern = limiter.get_pattern(key)

    pattern_icons = {
        RequestPattern.UNKNOWN: "[?]",
        RequestPattern.HUMAN_BURSTY: "[H]",
        RequestPattern.BOT_PERIODIC: "[B]",
        RequestPattern.BATCH_RAMP: "[R]",
        RequestPattern.BATCH_STEADY: "[S]",
    }

    icon = pattern_icons.get(pattern, "[?]")
    print(f"  {icon} {label}")
    print(f"      Pattern:     {pattern.name}")
    if analysis:
        print(f"      Confidence:  {analysis.confidence:.2f}")
        print(f"      CV:          {analysis.cv:.4f}")
        print(f"      Entropy:     {analysis.entropy:.3f}")
        print(f"      Autocorr:    {analysis.autocorrelation_peak:.3f} @ lag {analysis.autocorrelation_lag}")
        print(f"      Burst ratio: {analysis.burst_ratio:.3f}")
        print(f"      Trend slope: {analysis.trend_slope:.4f}")
    print(f"      Requests:    {stats['allowed']} allowed, {stats['denied']} denied")
    print()


def main():
    banner("Tempo: Adaptive Rate Limiter Demo")

    print("This demo simulates four different client behaviors against")
    print("the same adaptive rate limiter. Each client sends a similar")
    print("number of total requests, but with very different timing patterns.")
    print()
    print("The limiter detects each client's behavioral pattern and applies")
    print("different rate limiting policies accordingly.")
    print()

    # --- Demo 1: Lenient policy ---
    banner("Policy: LENIENT (base_rate=100/60s)")

    clock = ManualClock()
    config = DetectorConfig(min_samples=10)
    limiter = AdaptiveRateLimiter(
        policy_set=PolicySet.lenient(base_rate=100, window=60.0),
        detector_config=config,
        clock=clock,
        reclassify_interval=10,
    )

    # Run all simulations (each with its own key)
    bot_stats = simulate_bot(clock, limiter, "bot-scraper")
    human_stats = simulate_human(clock, limiter, "human-user")
    ramp_stats = simulate_batch_ramp(clock, limiter, "batch-import")
    steady_stats = simulate_batch_steady(clock, limiter, "bulk-processor")

    print("Results:")
    print()
    print_analysis(limiter, "bot-scraper", "Bot Scraper (periodic 500ms)", bot_stats)
    print_analysis(limiter, "human-user", "Human User (bursty clicks)", human_stats)
    print_analysis(limiter, "batch-import", "Batch Import (ramp up)", ramp_stats)
    print_analysis(limiter, "bulk-processor", "Bulk Processor (steady stream)", steady_stats)

    # --- Demo 2: Strict policy ---
    banner("Policy: STRICT (base_rate=60/60s)")

    clock2 = ManualClock()
    limiter2 = AdaptiveRateLimiter(
        policy_set=PolicySet.strict(base_rate=60, window=60.0),
        detector_config=config,
        clock=clock2,
        reclassify_interval=10,
    )

    bot_stats2 = simulate_bot(clock2, limiter2, "bot-scraper")
    human_stats2 = simulate_human(clock2, limiter2, "human-user")
    ramp_stats2 = simulate_batch_ramp(clock2, limiter2, "batch-import")
    steady_stats2 = simulate_batch_steady(clock2, limiter2, "bulk-processor")

    print("Results:")
    print()
    print_analysis(limiter2, "bot-scraper", "Bot Scraper (periodic 500ms)", bot_stats2)
    print_analysis(limiter2, "human-user", "Human User (bursty clicks)", human_stats2)
    print_analysis(limiter2, "batch-import", "Batch Import (ramp up)", ramp_stats2)
    print_analysis(limiter2, "bulk-processor", "Bulk Processor (steady stream)", steady_stats2)

    # --- Demo 3: Side-by-side comparison ---
    banner("Comparison: Same Client, Different Policies")

    print("  Bot Scraper:")
    print(f"    Lenient: {bot_stats['allowed']} allowed / {bot_stats['denied']} denied")
    print(f"    Strict:  {bot_stats2['allowed']} allowed / {bot_stats2['denied']} denied")
    print()
    print("  Human User:")
    print(f"    Lenient: {human_stats['allowed']} allowed / {human_stats['denied']} denied")
    print(f"    Strict:  {human_stats2['allowed']} allowed / {human_stats2['denied']} denied")
    print()

    # --- Demo 4: Seed numbers as request timestamps ---
    banner("Bonus: Seed Numbers as Request Timestamps")

    seeds = [7812, 5262, 2143, 424, 8528, 770, 4257, 6839, 4082, 3880, 8473, 2523, 8296, 2607, 9566, 9826]
    print(f"  Seed numbers: {seeds}")
    print()

    clock3 = ManualClock()
    config3 = DetectorConfig(min_samples=8)
    det_only = AdaptiveRateLimiter(
        detector_config=config3,
        clock=clock3,
        reclassify_interval=4,
    )
    sorted_seeds = sorted(seeds)
    for t in sorted_seeds:
        det_only.check("seeds", cost=1)
        clock3.set(float(t))

    analysis = det_only.get_analysis("seeds")
    if analysis:
        print(f"  Pattern detected: {analysis.pattern.name}")
        print(f"  Mean interval:    {analysis.mean_interval:.1f}")
        print(f"  CV:               {analysis.cv:.4f}")
        print(f"  Entropy:          {analysis.entropy:.3f}")
        print(f"  Autocorrelation:  {analysis.autocorrelation_peak:.3f}")
        print()
        print("  The seed numbers, treated as timestamps, form a pattern that")
        print("  the detector classifies based on their statistical properties.")

    banner("Demo Complete")


if __name__ == "__main__":
    main()

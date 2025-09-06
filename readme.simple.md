# Hamiltonian Neural Networks for Trading: A Simple Guide

## What is a Hamiltonian? Think of a Pendulum!

### The Pendulum on Your Grandfather Clock

Imagine a pendulum swinging back and forth on an old clock:

```
      O  (pivot)
     /|
    / |
   /  |           At the top: SLOW, lots of "stored up" energy
  *   |           (like a ball at the top of a hill)
      |
      |    *      In the middle: FAST, lots of "moving" energy
      |           (like a ball rolling at the bottom of a hill)
      |
      |  *        At the other side: SLOW again
      |           All the "moving" energy turned back into "stored" energy
```

Here is the magical thing: **the TOTAL energy never changes!**

```
At any moment:

  Moving Energy + Stored Energy = CONSTANT

  (Scientists call these "kinetic" and "potential" energy)
```

When the pendulum is at the top of its swing:
- Moving energy = LOW (it is barely moving)
- Stored energy = HIGH (gravity wants to pull it down)

When it swooshes through the bottom:
- Moving energy = HIGH (it is going fast!)
- Stored energy = LOW (it is at the lowest point)

But add them up and you always get the same number. Always. This is called **energy conservation**, and it was discovered by a mathematician named William Hamilton in the 1830s.

---

## How is a Market Like a Pendulum?

### The Price-Momentum Seesaw

Think about Bitcoin's price:

```
Phase 1: Price drops below average
         "Stored energy" builds up (like pulling the pendulum to one side)
         Bargain hunters start buying...

Phase 2: Buying momentum increases
         "Moving energy" builds (pendulum swinging through the middle)
         FOMO kicks in, everyone buys...

Phase 3: Price overshoots above average
         "Stored energy" builds up again (pendulum on other side)
         People start taking profits...

Phase 4: Selling momentum increases
         Price starts falling back...
         And the cycle repeats!
```

Let's draw it:

```
Price Deviation (how far from "normal")
    ^
    |        * <-- Price way too high (stored energy)
    |       / \
    |      /   \     * <-- Overshoots again
    |     /     \   / \
----+----/-------\-/---\----> Time
    |  /          *     \
    | /                  \
    |*                    * <-- Price way too low (stored energy)
    |
```

This looks EXACTLY like a pendulum swinging!

```
Pendulum:                    Market:
Angle (position) ------->   Price deviation (how far from average)
Speed (momentum) ------->   Rate of price change (momentum)
Gravity ---------->          Mean-reversion force (pushes price back to average)
Air resistance ---->         Transaction costs (slows things down)
```

---

## What is a Phase Portrait?

If you track both the position AND the speed of the pendulum at every moment, and plot them on a chart, you get something beautiful:

```
    Speed (momentum)
        ^
        |     .---.
        |    /     \
        |   |   .   |     <-- Each loop is one full swing
        |    \     /
        |     '---'
        +----------------> Position (displacement)
```

For a market:

```
    Price Momentum (how fast price is changing)
        ^
        |      .---.
        |     /     \
        |    |   .   |    <-- One market cycle
        |     \     /
        |      '---'
        +-------------------> Price Deviation (how far from normal)
```

If the market is like a perfect pendulum, these loops would be perfect circles or ellipses. In reality, they spiral inward (because of friction/costs) or sometimes jump to new orbits (regime changes).

---

## What is a Hamiltonian Neural Network?

### Regular Neural Networks vs HNN

**Regular neural network approach:**
"Hey neural network, here is some price data. Just predict what happens next. Figure it out yourself."

The problem: the network has NO idea about energy conservation. Its predictions can drift off into nonsense over time.

```
Regular NN prediction after many steps:

  Price might do:     What actually happens:
      /                    /\    /\
     /                    /  \  /  \
    /   (drift to         /    \/    \
   /     infinity!)
```

**Hamiltonian Neural Network approach:**
"Hey neural network, learn the ENERGY function of this system. I know total energy is (approximately) conserved."

```
HNN prediction after many steps:

  What HNN predicts:     What actually happens:
      /\    /\               /\    /\
     /  \  /  \             /  \  /  \
    /    \/    \            /    \/    \
   (stays bounded!)
```

### How Does It Work?

```
Step 1: Feed price data (position) and momentum data into a neural network
        ┌──────────────────────────┐
  q --> │                          │
        │    Neural Network        │ --> H (one number: "energy")
  p --> │                          │
        └──────────────────────────┘

Step 2: Use calculus (automatic differentiation) to get the RULES OF MOTION
        from the energy function

        dH/dp  = how position changes  (velocity)
       -dH/dq  = how momentum changes  (force)

Step 3: Use these rules to predict the future!
        Starting from today's (position, momentum), step forward in time
        following the rules. The predictions naturally conserve energy.
```

It is like the network learns the LAWS OF PHYSICS of the market, not just a pattern to memorize.

---

## Why Is This Better Than Regular Prediction?

### The Bowling Ball Analogy

Imagine you want to predict where a bowling ball will roll:

**Method A (Regular NN):** Watch 1000 bowling balls roll. Memorize every path. Hope the next ball does something similar.

**Method B (HNN):** Learn that balls obey gravity and friction. From those LAWS, predict where ANY ball will roll, even on a lane you have never seen.

Method B is more powerful because it understands the RULES, not just examples.

For markets:
- **Method A** breaks down when the market enters a new regime
- **Method B** still works because the underlying energy dynamics are similar

---

## The Dissipative Extension: Markets Have Friction!

A real pendulum eventually stops because of air resistance. Markets have their own "air resistance":

```
Perfect pendulum (no friction):     Real pendulum (with friction):

    Speed                               Speed
      ^                                   ^
      |  .---.                            |  .---.
      | |     |                           | |  .--.
      | |  .  |  (loops forever)          | | | .. |  (spirals inward)
      | |     |                           | |  '--'
      |  '---'                            |  '---'
      +---------> Position                +---------> Position
```

For markets, the "friction" comes from:
- Transaction costs (every trade costs money)
- Bid-ask spread (you always buy high and sell low)
- Information decay (old patterns fade)
- Market impact (big trades move the price against you)

The **Dissipative HNN** adds a friction term:

```
Energy change = -friction * speed

Instead of: total energy = constant
We get:     total energy slowly decreases (like a real market)
```

This makes the model much more realistic!

---

## A Real-World Example: Trading Bitcoin on Bybit

### Step 1: Get the Data

We download Bitcoin price and volume data from Bybit exchange.

### Step 2: Create the "Phase Space"

```
From raw data:              To phase space:

  Price: $50,000              q = how far price is from its 20-period average
  Volume: 1000 BTC            p = how fast price is changing (momentum)
  Time: 10:00 AM
```

### Step 3: Train the HNN

The neural network learns the energy function H(q, p). After training, it knows:
- How much "energy" the market has right now
- How momentum and price deviation trade off
- What the natural oscillation period is

### Step 4: Make Predictions

```
Current state: q = +0.02 (price 2% above average), p = -0.01 (falling)

HNN predicts: "Energy is being converted from 'stored' (price deviation)
               to 'kinetic' (momentum). Price will continue falling
               toward the average, overshoot slightly, then bounce back."

Trading signal: SELL (price will fall toward average)
```

### Step 5: Risk Management

```
If the "energy" suddenly jumps (like the pendulum getting kicked):

  Energy
    |          * <-- Sudden jump! Something changed!
    |     ****
    | ****
    |*
    +-----------> Time

Signal: "REGIME CHANGE - reduce positions!"
```

This is one of the coolest features of HNN -- it can detect when the market's "physics" change.

---

## What Makes This Special?

```
                    Regular NN        HNN
                    ─────────         ───
Long predictions:   Drift away        Stay bounded
Physics:            None              Energy conservation
Interpretability:   Black box         "The market has X energy"
Regime detection:   Difficult         Natural (energy jumps)
Data efficiency:    Need lots         Works with less data
                                      (physics helps fill gaps)
```

---

## Summary

1. **Markets oscillate** like pendulums -- price and momentum trade off
2. **Hamiltonian Neural Networks** learn the "energy function" of the market
3. **Energy conservation** keeps predictions stable over long horizons
4. **Dissipation** handles real-world friction (costs, spreads)
5. **Phase portraits** give a beautiful visual understanding of market dynamics
6. **Energy anomalies** detect regime changes naturally

Think of it this way: instead of trying to predict exactly where every wave on the ocean will be, HNNs learn the RULES that govern how waves work. That is a much more powerful way to understand the market.

---

*Chapter 149 of Machine Learning for Trading. Simple explanation for beginners.*

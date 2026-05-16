# Viterbi Algorithm for Gene Structure Prediction
### Hidden Markov Model — Bioinformatics Assignment

---

## Table of Contents

1. [Overview](#overview)
2. [Biological Background](#biological-background)
3. [HMM Model Description](#hmm-model-description)
4. [Algorithm Explanation](#algorithm-explanation)
5. [File Structure](#file-structure)
6. [Dependencies & Setup](#dependencies--setup)
7. [How to Run](#how-to-run)
8. [Code Walkthrough](#code-walkthrough)
9. [Output Explanation](#output-explanation)
10. [Results & Interpretation](#results--interpretation)

---

## Overview

This assignment implements the **Viterbi Algorithm** on a **Hidden Markov Model (HMM)** to predict gene structure — specifically, to locate the **5' splice site** that separates an exon region from an intron region within a DNA sequence.

Given the query sequence:

```
CTTCATGTGAAAGCAGACGTAAGTCA
```

the algorithm finds the most probable sequence of hidden states (Exon / 5'ss / Intron) that could have generated it.

---

## Biological Background

In eukaryotic genes, pre-mRNA is processed by **splicing** — the removal of non-coding **introns** and joining of coding **exons**. The boundary between an exon and an intron is called the **5' splice site (5'ss)**, which is characterised by the consensus dinucleotide **GU** (or **GT** in DNA).

This HMM models that biology:

```
Start (s)  →  Exon (E)  →  5'ss (5)  →  Intron (I)  →  End (e)
```

---

## HMM Model Description

### States

| Symbol | Index | Description |
|--------|-------|-------------|
| `s` | 0 | Silent **start** state |
| `E` | 1 | **Exon** — coding region |
| `5` | 2 | **5' splice site** — single-nucleotide boundary |
| `I` | 3 | **Intron** — non-coding region |
| `e` | 4 | Silent **end** state |

### State Transition Probabilities

```
         s     E     5     I     e
    s  [ 0.0   1.0   0.0   0.0   0.0 ]
    E  [ 0.0   0.9   0.1   0.0   0.0 ]
    5  [ 0.0   0.0   0.0   1.0   0.0 ]
    I  [ 0.0   0.0   0.0   0.9   0.1 ]
    e  [ 0.0   0.0   0.0   0.0   0.0 ]
```

- `s` always enters the Exon state
- Exon self-loops with probability 0.9; transitions to 5'ss with probability 0.1
- 5'ss always transitions to Intron
- Intron self-loops with probability 0.9; terminates with probability 0.1

### Emission Probabilities

```
         A      C      G      T
    s  [ 0.00   0.00   0.00   0.00 ]   ← silent
    E  [ 0.25   0.25   0.25   0.25 ]   ← uniform (any nucleotide equally likely)
    5  [ 0.05   0.00   0.95   0.00 ]   ← strongly prefers G
    I  [ 0.40   0.10   0.10   0.40 ]   ← AT-rich
    e  [ 0.00   0.00   0.00   0.00 ]   ← silent
```

---

## Algorithm Explanation

### Viterbi Algorithm (Dynamic Programming)

The Viterbi algorithm finds the single most probable hidden state sequence for a given observation sequence. It avoids the exponential cost of evaluating every possible path by using dynamic programming.

#### Notation

| Symbol | Meaning |
|--------|---------|
| `V[j][t]` | Best log-probability of any path ending in state `j` after emitting `seq[0..t]` |
| `T[j][t]` | Index of the previous state on that best path (used for traceback) |
| `a(i→j)` | Transition probability from state `i` to state `j` |
| `b(j, x)` | Emission probability of nucleotide `x` from state `j` |

#### Step 1 — Initialisation (t = 0)

```
V[j][0] = log( a(s → j) ) + log( b(j, seq[0]) )   for each state j
T[j][0] = 0   (all paths come from the start state)
```

Since `s` can only go to `E`, only `V[E][0]` is finite (`= log(0.25)`).

#### Step 2 — Recursion (t = 1 … L−1)

```
V[j][t] = max over all i of:  V[i][t-1] + log( a(i→j) ) + log( b(j, seq[t]) )
T[j][t] = argmax i of the above
```

All arithmetic is done in **log space** to avoid floating-point underflow.

#### Step 3 — Traceback

```
1. Find  best_last = argmax_j  V[j][L-1]
2. For t = L-1 down to 1:
       state[t-1] = T[ state[t] ][ t ]
3. Reverse the collected indices → optimal state path
```

---

## File Structure

```
.
├── viterbi_full.py          # Complete, runnable Python implementation
├── viterbi_completed.ipynb  # Jupyter notebook with step-by-step cells
└── README.md                # This file
```

---

## Dependencies & Setup

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python  | ≥ 3.8   | Runtime |
| NumPy   | ≥ 1.19  | Matrix operations |

### Installation

```bash
# Using pip
pip install numpy

# Using conda
conda install numpy
```

No other third-party libraries are required.

---

## How to Run

### Option 1 — Python Script

```bash
python viterbi_full.py
```

### Option 2 — Jupyter Notebook

```bash
jupyter notebook viterbi_completed.ipynb
```

Run cells from top to bottom in order.

---

## Code Walkthrough

### `safe_log(x)`

```python
def safe_log(x):
    return math.log(x) if x > 0 else float('-inf')
```

Wraps `math.log` to return `-inf` instead of raising a domain error when `x = 0`. Essential because many transitions and emissions are zero.

---

### `get_log_prob_for_state_path(state_path, sequence)`

Computes the log-probability of a **manually specified** state path over the sequence. Used in the assignment to compare fixed candidate paths (k1 through k6) before running Viterbi.

```python
res = math.log(0.25)   # first nucleotide always emitted from state E
for i in range(1, len(state_path)):
    res += math.log( trans[path[i-1]][path[i]] * emis[path[i]][seq[i]] )
```

---

### Matrix Initialisation

```python
viterbi_value_matrix = np.full((n_states, n_obs), -inf)
viterbi_trace_matrix = np.zeros((n_states, n_obs), dtype=int)

for j in range(n_states):
    V[j][0] = safe_log(trans[s][j]) + safe_log(emis[j][seq[0]])
    T[j][0] = 0
```

---

### `calculate_prob_for_a_node(j, t)`

The core Viterbi cell computation:

```python
def calculate_prob_for_a_node(j, t):
    log_emis = safe_log(emis[j][seq[t]])
    if log_emis == -inf:
        return -inf, 0

    max_val, best_prev = -inf, 0
    for i in range(n_states):
        if trans[i][j] > 0 and V[i][t-1] != -inf:
            val = V[i][t-1] + log(trans[i][j]) + log_emis
            if val > max_val:
                max_val, best_prev = val, i

    return max_val, best_prev
```

Returns the best log-probability **and** the index of the predecessor state.

---

### Recursion Loop

```python
for t in range(1, n_obs):
    for j in range(n_states):
        V[j][t], T[j][t] = calculate_prob_for_a_node(j, t)
```

Fills the entire `n_states × n_obs` matrix left to right.

---

### `traceback_viterbi()`

```python
def traceback_viterbi():
    best_last = argmax( V[:, n_obs-1] )
    path = [best_last]
    for t in range(n_obs-1, 0, -1):
        path.append( T[path[-1]][t] )
    path.reverse()
    return ''.join(id2state[s] for s in path)
```

---

## Output Explanation

Running the script produces three sections:

### 1. Reference Log-Probabilities

Log-probabilities of the six manually defined candidate paths. These are computed before Viterbi and show which fixed path is best among those tested.

### 2. Viterbi Value Matrix

```
         C       T       T  ...
s     -inf    -inf    -inf  ...
E   -1.386  -2.878  -4.370  ...
5     -inf    -inf    -inf  ...
I     -inf    -inf    -inf  ...
e     -inf    -inf    -inf  ...
```

Each cell `V[j][t]` holds the log-probability of the best path ending at state `j`, position `t`. `-inf` means the state is unreachable at that position given the model constraints.

### 3. Viterbi Trace Matrix

Records which previous state `i` produced the best value at each cell — used during traceback.

---

## Results & Interpretation

```
Sequence  : CTTCATGTGAAAGCAGACGTAAGTCA
Best path : EEEEEEEEEEEEEEEEEEEEEEEEEE
Log-prob  : −38.68
```

| Path | Log-probability |
|------|----------------|
| k1 — 5'ss @ pos 6  | −43.90 |
| k2 — 5'ss @ pos 8  | −43.45 |
| k3 — 5'ss @ pos 12 | −43.94 |
| k4 — 5'ss @ pos 15 | −42.58 |
| k5 — 5'ss @ pos 18 | −41.22 |
| k6 — 5'ss @ pos 22 | −41.71 |
| only E             | −40.98 |
| **Viterbi (optimal)** | **−38.68** |

### Why is the all-Exon path optimal?

The 5'ss state (state `5`) emits G with probability 0.95 but emits C and T with only 0.05 and 0.00 respectively. Transitioning through `5` incurs a high penalty unless the nucleotide at that position is strongly G. While there are G nucleotides in the sequence, the combined cost of the `E→5` transition (probability 0.1) plus the weak intron AT-rich signal means the Exon self-loop (probability 0.9, uniform emission) is always more competitive on this particular sequence. The Viterbi algorithm — unlike the fixed candidate paths which include an artificial `+log(0.1)` end-state penalty — finds the globally optimal path without that constraint, yielding the best achievable log-probability of **−38.68**.

---

*Assignment: Hidden Markov Models & the Viterbi Algorithm — Bioinformatics*

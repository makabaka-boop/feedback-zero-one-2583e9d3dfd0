#!/usr/bin/env python3
"""phylo_backend.py -- ascertainment-bias-corrected phylogenetic backend.

Model
-----
Rooted tree (>= 2 leaves), two states A/C.  Per site, a rate r is drawn once
from the discrete mixture ``rates`` (probability w) and shared by the whole
tree; the root state follows ``pi``; given the parent state each branch
evolves independently as a CTMC with transition matrix exp(r*t*Q).

For each site let
  V = the true leaf states are not all identical,
  E = the observation ("A"/"C"/"?" per leaf, "?" allows both) is compatible.
Output per site:
  likelihood = P(E and V) / P(V)
  counts     = expected number of jumps (state changes, either direction)
               on each edge, conditional on (E and V), in input edge order.
Joint probabilities are mixed over rates first, then conditioned.

Usage
-----
    python3 phylo_backend.py INPUT.json OUTPUT.json

INPUT.json is an object with exactly the members "edges", "pi", "Q", "rates",
"sites".  Every number is a string "p/q" with integer p and positive integer q.

  edges : [[parent, child, t], ...]   non-empty ASCII alphanumeric identifiers,
                                      a single rooted tree, >= 2 leaves, t >= 0
  pi    : [pA, pC]                    probabilities summing to 1
  Q     : [[qAA, qAC], [qCA, qCC]]    off-diagonal >= 0, row sums 0
  rates : [[r, w], ...]               non-empty, r >= 0, w > 0, sum w = 1
  sites : [{leaf: "A"|"C"|"?", ...}, ...]   each map covers exactly all leaves

Output (written to OUTPUT.json, replacing it, only on success):
  []                              if sites is empty (takes precedence)
  {"status": "unobservable"}      if P(V) = 0
  [ {"likelihood": x, "counts": [...]}, {"status": "impossible"}, ... ]
                                  per site in input order; sites whose
                                  numerator P(E and V) is zero yield only
                                  {"status": "impossible"}

Alignment import entry
----------------------
The same CLI also accepts an input object where "sequences" replaces "sites"
(the two may not coexist; all other members are unchanged).  Only trees with
2 to 4 leaves are accepted then, and "sequences" must map exactly all leaves
to (possibly empty) A/C strings, e.g. {"a": "A", "b": "CA"}.

The leaf sequences are aligned by inserting gaps "-" into equal-length rows
(gap removal restores each original string; all-gap columns are forbidden).
The total cost summed over all leaf pairs is minimized: per pair, columns
where both rows show a gap are deleted first, identical letters cost 0,
different letters 1, and a maximal run of k consecutive gaps on the same
side costs k + 1 (switching sides starts a new run; deleted columns neither
cost anything nor break a run).  Ties are broken by the fewest columns, then
by the lexicographically smallest column-major expansion of the alignment
(leaves in ASCII order per column, character order "-" < "A" < "C").

The output contains exactly two members:
  alignment : {leaf: aligned row, ...}
  result    : the original result (same type and precision) obtained by
              feeding every alignment column to the site inference with
              gaps converted to "?"; all columns are kept.  When every
              sequence is empty all rows are "" and result is [].

On any invalid input (including duplicate or unknown members) the whole run
is rejected: an error object {"status": "error", "path": <JSON path>,
"index": <zero-based index or null>, "message": ...} is printed to stdout,
the exit code is 1 and a pre-existing OUTPUT.json is kept untouched.
JSON syntax errors are reported with path "$".
"""

import itertools
import json
import math
import os
import re
import sys
from fractions import Fraction


# ---------------------------------------------------------------------------
# Rejection handling
# ---------------------------------------------------------------------------

class Reject(Exception):
    """Input rejection carrying a JSON path and a zero-based index (or None)."""

    def __init__(self, path, message, index=None):
        super().__init__(message)
        self.path = path
        self.message = message
        self.index = index


def reject(path, message, index=None):
    raise Reject(path, message, index)


class Pairs(list):
    """Members of a JSON object as (key, value) pairs (duplicates preserved)."""


def _is_array(value):
    return isinstance(value, list) and not isinstance(value, Pairs)


def _as_object(value, path, index=None):
    if not isinstance(value, Pairs):
        reject(path, "expected a JSON object", index)
    obj = {}
    for key, val in value:
        if key in obj:
            reject(path, "duplicate member %r" % (key,), index)
        obj[key] = val
    return obj


IDENT_RE = re.compile(r"[A-Za-z0-9]+")
FRACTION_RE = re.compile(r"([+-]?[0-9]+)/([0-9]+)")


def _as_fraction(value, path, index=None):
    """Parse a number encoded as a "p/q" string into an exact Fraction."""
    if not isinstance(value, str):
        reject(path, 'expected a number encoded as a "p/q" string', index)
    m = FRACTION_RE.fullmatch(value)
    if m is None:
        reject(path, 'expected a number encoded as a "p/q" string', index)
    q = int(m.group(2))
    if q == 0:
        reject(path, "denominator q must be a positive integer", index)
    return Fraction(int(m.group(1)), q)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_edges(value):
    path = "$.edges"
    if not _is_array(value):
        reject(path, "edges must be an array of [parent, child, t] triples")
    edges = []
    nodes = set()
    parent_of = {}
    for i, entry in enumerate(value):
        epath = "%s[%d]" % (path, i)
        if not _is_array(entry) or len(entry) != 3:
            reject(epath, "edge must be a [parent, child, t] triple", i)
        parent, child, t = entry
        for k, name in ((0, parent), (1, child)):
            if not isinstance(name, str) or IDENT_RE.fullmatch(name) is None:
                reject("%s[%d]" % (epath, k),
                       "node identifier must be a non-empty ASCII "
                       "alphanumeric string", i)
        if parent == child:
            reject(epath, "self-loops are not allowed in a tree", i)
        ft = _as_fraction(t, "%s[2]" % epath, i)
        if ft < 0:
            reject("%s[2]" % epath, "branch length t must be >= 0", i)
        if child in parent_of:
            reject("%s[1]" % epath,
                   "node %r has more than one parent (duplicate child)"
                   % (child,), i)
        parent_of[child] = parent
        nodes.add(parent)
        nodes.add(child)
        edges.append((parent, child, float(ft)))
    roots = sorted(v for v in nodes if v not in parent_of)
    if len(roots) != 1:
        reject(path, "edges must form a single rooted tree "
                     "(found %d candidate root(s))" % len(roots))
    root = roots[0]
    children = {}
    for parent, child, _t in edges:
        children.setdefault(parent, []).append(child)
    seen = set()
    stack = [root]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(children.get(v, ()))
    if seen != nodes:
        reject(path, "edges must form a single connected rooted tree")
    leaves = sorted(v for v in nodes if v not in children)
    if len(leaves) < 2:
        reject(path, "the rooted tree must have at least two leaves")
    return edges, root, leaves


def check_pi(value):
    path = "$.pi"
    if not _is_array(value) or len(value) != 2:
        reject(path, "pi must be an array of two probabilities")
    probs = []
    for i, v in enumerate(value):
        f = _as_fraction(v, "%s[%d]" % (path, i), i)
        if f < 0:
            reject("%s[%d]" % (path, i), "probabilities must be >= 0", i)
        probs.append(f)
    if probs[0] + probs[1] != 1:
        reject(path, "pi must sum to 1")
    return [float(probs[0]), float(probs[1])]


def check_Q(value):
    path = "$.Q"
    if not _is_array(value) or len(value) != 2:
        reject(path, "Q must be a 2x2 matrix")
    Q = []
    for i, row in enumerate(value):
        if not _is_array(row) or len(row) != 2:
            reject("%s[%d]" % (path, i), "each row of Q must have two entries", i)
        Q.append([_as_fraction(row[0], "%s[%d][0]" % (path, i), i),
                  _as_fraction(row[1], "%s[%d][1]" % (path, i), i)])
    if Q[0][1] < 0:
        reject("$.Q[0][1]", "off-diagonal entries of Q must be >= 0", 0)
    if Q[1][0] < 0:
        reject("$.Q[1][0]", "off-diagonal entries of Q must be >= 0", 1)
    if Q[0][0] + Q[0][1] != 0:
        reject("$.Q[0]", "row sums of Q must be 0", 0)
    if Q[1][0] + Q[1][1] != 0:
        reject("$.Q[1]", "row sums of Q must be 0", 1)
    # Q = [[-a, a], [b, -b]] with a, b >= 0
    return float(Q[0][1]), float(Q[1][0])


def check_rates(value):
    path = "$.rates"
    if not _is_array(value) or len(value) == 0:
        reject(path, "rates must be a non-empty array of [r, w] pairs")
    rates = []
    wsum = Fraction(0)
    for i, entry in enumerate(value):
        epath = "%s[%d]" % (path, i)
        if not _is_array(entry) or len(entry) != 2:
            reject(epath, "rate entry must be an [r, w] pair", i)
        r = _as_fraction(entry[0], "%s[0]" % epath, i)
        w = _as_fraction(entry[1], "%s[1]" % epath, i)
        if r < 0:
            reject("%s[0]" % epath, "rate r must be >= 0", i)
        if w <= 0:
            reject("%s[1]" % epath, "rate weight w must be > 0", i)
        wsum += w
        rates.append((float(r), float(w)))
    if wsum != 1:
        reject(path, "rate weights must sum to 1")
    return rates


def check_sites(value, leaf_set):
    path = "$.sites"
    if not _is_array(value):
        reject(path, "sites must be an array of leaf-state objects")
    sites = []
    for i, entry in enumerate(value):
        epath = "%s[%d]" % (path, i)
        site = _as_object(entry, epath, i)
        keys = set(site)
        if keys != leaf_set:
            unknown = sorted(keys - leaf_set)
            missing = sorted(leaf_set - keys)
            if unknown:
                reject(epath, "unknown leaf member(s): " + ", ".join(unknown), i)
            reject(epath, "missing leaf member(s): " + ", ".join(missing), i)
        obs = {}
        for name, state in site.items():
            if state not in ("A", "C", "?"):
                reject("%s.%s" % (epath, name),
                       'site values must be "A", "C" or "?"', i)
            obs[name] = state
        sites.append(obs)
    return sites


SEQUENCE_RE = re.compile(r"[AC]*")


def check_sequences(value, leaf_set):
    path = "$.sequences"
    seqs = _as_object(value, path)
    keys = set(seqs)
    if keys != leaf_set:
        unknown = sorted(keys - leaf_set)
        missing = sorted(leaf_set - keys)
        if unknown:
            reject(path, "unknown leaf member(s): " + ", ".join(unknown))
        reject(path, "missing leaf member(s): " + ", ".join(missing))
    out = {}
    for name, seq in seqs.items():
        if not isinstance(seq, str) or SEQUENCE_RE.fullmatch(seq) is None:
            reject("%s.%s" % (path, name),
                   'sequence values must be strings over "A" and "C" '
                   "(possibly empty)")
        out[name] = seq
    return out


def parse_input(text):
    def bad_constant(x):
        raise ValueError("invalid constant %s" % x)

    try:
        data = json.loads(text, object_pairs_hook=Pairs,
                          parse_constant=bad_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        reject("$", "invalid JSON syntax (%s)" % exc)
    top = _as_object(data, "$")
    # "sequences" replaces "sites" for the alignment import entry; the two
    # may not coexist (with both present, "sequences" is an unknown member
    # of the original entry).
    sequences_mode = "sequences" in top and "sites" not in top
    required = ("edges", "pi", "Q", "rates",
                "sequences" if sequences_mode else "sites")
    for key in top:
        if key not in required:
            reject("$." + key, "unknown top-level member %r" % (key,))
    for key in required:
        if key not in top:
            reject("$", "missing top-level member %r" % (key,))
    edges, root, leaves = check_edges(top["edges"])
    if sequences_mode and len(leaves) > 4:
        reject("$.edges", "the sequences entry accepts only trees with "
                          "2 to 4 leaves")
    pi = check_pi(top["pi"])
    a_q, b_q = check_Q(top["Q"])
    rates = check_rates(top["rates"])
    model = {"edges": edges, "root": root, "leaves": leaves, "pi": pi,
             "a_q": a_q, "b_q": b_q, "rates": rates}
    if sequences_mode:
        model["sequences"] = check_sequences(top["sequences"], set(leaves))
        return "sequences", model
    model["sites"] = check_sites(top["sites"], set(leaves))
    return "sites", model


# ---------------------------------------------------------------------------
# Two-state CTMC: transition probabilities and jump-count weights
# ---------------------------------------------------------------------------

def transition_and_jumps(a, b, t):
    """For Q = [[-a, a], [b, -b]] and time t return (P, J) where
    P = exp(t*Q) and J[i][j] = E[#jumps in [0, t] * 1(X_t = j) | X_0 = i].

    J = integral_0^t exp(Q s) R exp(Q (t-s)) ds with R the off-diagonal
    part of Q.  Closed form via exp(Q s) = P0 + exp(-s*(a+b)) * (I - P0)
    with P0 the stationary projection; the result satisfies the reversible
    identity pi_0 * J[0][1] = pi_1 * J[1][0] and row sums equal to the
    unconditional expected jump counts.
    """
    s = a + b
    if s <= 0.0 or t <= 0.0:
        return ((1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (0.0, 0.0))
    u = math.exp(-s * t)
    P = (((b + a * u) / s, a * (1.0 - u) / s),
         (b * (1.0 - u) / s, (a + b * u) / s))
    ab = a * b
    s2 = s * s
    t1 = (1.0 - u) / s   # integral of exp(-s*x) over [0, t]
    t2 = t * u           # integral of exp(-s*t) over [0, t]
    k = 2.0 * ab / s2
    d = a - b
    J = ((k * (b * t + d * t1 - a * t2),
          k * a * (t + t2) + a * d * d * t1 / s2),
         (k * b * (t + t2) + b * d * d * t1 / s2,
          k * (a * t - d * t1 - b * t2)))
    return P, J


# ---------------------------------------------------------------------------
# Likelihood / counts computation
# ---------------------------------------------------------------------------

def solve(edges, root, leaves, pi, a_q, b_q, rates, sites):
    n_edges = len(edges)
    children = {}
    for ei, (parent, child, _t) in enumerate(edges):
        children.setdefault(parent, []).append((child, ei))

    # Post-order (children before parents) and its reverse (parents first).
    postorder = []
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            postorder.append(node)
        else:
            stack.append((node, True))
            for child, _ei in children.get(node, ()):
                stack.append((child, False))
    parents_first = postorder[::-1]

    # Per-rate branch matrices.
    rate_data = []
    for r, w in rates:
        mats = [transition_and_jumps(r * a_q, r * b_q, t)
                for (_p, _c, t) in edges]
        rate_data.append((w, [m[0] for m in mats], [m[1] for m in mats]))

    def upward(site, ps):
        """Pruning pass.  Returns per-node triples (A, C, M) of partial
        likelihoods: subtree compatible with the observation AND its leaves
        all-A (A), all-C (C) or not-all-identical (M).  The mixed category
        is accumulated with non-negative terms only, so exact zeros are
        preserved (needed to detect P(V) = 0)."""
        A = {}
        C = {}
        M = {}
        for v in postorder:
            ch = children.get(v)
            if not ch:
                obs = "?" if site is None else site[v]
                if obs == "A":
                    A[v] = (1.0, 0.0); C[v] = (0.0, 0.0); M[v] = (0.0, 0.0)
                elif obs == "C":
                    A[v] = (0.0, 0.0); C[v] = (0.0, 1.0); M[v] = (0.0, 0.0)
                else:  # "?"
                    A[v] = (1.0, 0.0); C[v] = (0.0, 1.0); M[v] = (0.0, 0.0)
                continue
            av = [0.0, 0.0]
            cv = [0.0, 0.0]
            mv = [0.0, 0.0]
            for st in (0, 1):
                first = True
                a_run = c_run = m_run = 0.0
                for child, ei in ch:
                    P = ps[ei]
                    ac = A[child]
                    cc = C[child]
                    mc = M[child]
                    al = P[st][0] * ac[0] + P[st][1] * ac[1]
                    ga = P[st][0] * cc[0] + P[st][1] * cc[1]
                    mu = P[st][0] * mc[0] + P[st][1] * mc[1]
                    if first:
                        a_run, c_run, m_run = al, ga, mu
                        first = False
                    else:
                        m_run = (m_run * (al + ga + mu)
                                 + a_run * (ga + mu)
                                 + c_run * (al + mu))
                        a_run *= al
                        c_run *= ga
                av[st] = a_run
                cv[st] = c_run
                mv[st] = m_run
            A[v] = av
            C[v] = cv
            M[v] = mv
        return A, C, M

    def downward(L, ps, js):
        """Root-to-leaves pass: for each edge e = (u -> v) accumulate
        S[e] = sum_{i,j} P(X_u=i, data outside v) * J_e[i][j] * L_v[j],
        i.e. E[#jumps on e * 1(data)] for the leaf-likelihood scenario L."""
        S = [0.0] * n_edges
        g = {root: (pi[0], pi[1])}
        for u in parents_first:
            ch = children.get(u)
            if not ch:
                continue
            bs = []
            for child, ei in ch:
                P = ps[ei]
                lc = L[child]
                bs.append((P[0][0] * lc[0] + P[0][1] * lc[1],
                           P[1][0] * lc[0] + P[1][1] * lc[1]))
            gu = g[u]
            for k, (child, ei) in enumerate(ch):
                u0, u1 = gu
                for l in range(len(ch)):
                    if l != k:
                        u0 *= bs[l][0]
                        u1 *= bs[l][1]
                J = js[ei]
                lc = L[child]
                S[ei] += (u0 * (J[0][0] * lc[0] + J[0][1] * lc[1])
                          + u1 * (J[1][0] * lc[0] + J[1][1] * lc[1]))
                P = ps[ei]
                g[child] = (u0 * P[0][0] + u1 * P[1][0],
                            u0 * P[0][1] + u1 * P[1][1])
        return S

    # Valid empty sites takes precedence.
    if not sites:
        return []

    # P(V) is site-independent (observations ignored).
    p_v = 0.0
    for w, ps, _js in rate_data:
        _A, _C, M = upward(None, ps)
        p_v += w * (pi[0] * M[root][0] + pi[1] * M[root][1])
    if not p_v > 0.0:
        return {"status": "unobservable"}

    output = []
    for site in sites:
        p_ev = 0.0          # mixture of P_r(E and V)
        num = [0.0] * n_edges  # mixture of E_r[#jumps on e * 1(E and V)]
        for w, ps, js in rate_data:
            A, C, M = upward(site, ps)
            p_ev += w * (pi[0] * M[root][0] + pi[1] * M[root][1])
            L_obs = {}
            for v in A:
                av, cv, mv = A[v], C[v], M[v]
                L_obs[v] = (av[0] + cv[0] + mv[0], av[1] + cv[1] + mv[1])
            s_obs = downward(L_obs, ps, js)
            s_all_a = downward(A, ps, js)
            s_all_c = downward(C, ps, js)
            for ei in range(n_edges):
                num[ei] += w * (s_obs[ei] - s_all_a[ei] - s_all_c[ei])
        if not p_ev > 0.0:
            output.append({"status": "impossible"})
        else:
            output.append({
                "likelihood": p_ev / p_v,
                "counts": [x / p_ev for x in num],
            })
    return output


# ---------------------------------------------------------------------------
# Alignment core (sequences entry)
# ---------------------------------------------------------------------------
#
# Sum-of-pairs alignment of 2-4 sequences over {A, C}.  A column is a
# non-empty subset of the sequences contributing their next letter (the
# other rows show "-"); all-gap columns therefore never occur and removing
# gaps restores each original string.  The pairwise cost of two aligned
# rows is computed after deleting the columns where both rows show a gap:
# identical letters cost 0, different letters 1, and a maximal run of k
# consecutive columns with the gap on the same side costs k + 1 (switching
# sides starts a new run; deleted columns neither cost anything nor break
# a run).  The total cost is the sum over all leaf pairs; ties are broken
# by the fewest columns, then by the lexicographically smallest
# column-major expansion (leaves in ASCII order, "-" < "A" < "C").
#
# The pairwise cost is history-dependent: a gap run stays open across
# columns in which the pair is not involved.  The dynamic program below
# therefore tracks, besides the consumed prefix of each sequence, the
# total preorder of the sequences' last advancement columns: for a pair
# (i, j) the last relevant column showed a gap in j iff i advanced more
# recently than j.  There are only 3 / 13 / 75 total preorders on 2 / 3 /
# 4 sequences, so an exact DAG shortest-path search over (position,
# preorder) states is cheap for the short fragments accepted here.

def _normalize_preorder(c):
    """Canonical rank tuple of a total preorder (classes numbered 0..k-1
    from oldest to most recent)."""
    ranks = {v: k for k, v in enumerate(sorted(set(c)))}
    return tuple(ranks[v] for v in c)


def _preorder_tables(n):
    """(preorders, initial index, transition table, gap-cost table) for
    n sequences.  transition[c][mask] is the new preorder index after a
    column in which exactly the sequences in `mask` advance; gapcost[c]
    [mask] is the gap extension + opening cost of that column summed over
    all pairs (letter mismatches are added separately)."""
    preorders = sorted({_normalize_preorder(c)
                        for c in itertools.product(range(n), repeat=n)})
    index = {c: k for k, c in enumerate(preorders)}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    trans = []
    gapcost = []
    for c in preorders:
        top = max(c) + 1
        trow = [0] * (1 << n)
        grow = [0] * (1 << n)
        for mask in range(1, 1 << n):
            trow[mask] = index[_normalize_preorder(tuple(
                top if (mask >> i) & 1 else c[i] for i in range(n)))]
            g = 0
            for i, j in pairs:
                bi = (mask >> i) & 1
                bj = (mask >> j) & 1
                if bi == bj:
                    continue
                g += 1                    # one gap column for the pair
                if bi:
                    # Gap in j; the run continues iff i advanced strictly
                    # more recently than j, otherwise a new run opens.
                    if c[i] <= c[j]:
                        g += 1
                elif c[j] <= c[i]:        # gap in i
                    g += 1
            grow[mask] = g
        trans.append(trow)
        gapcost.append(grow)
    return preorders, index[(0,) * n], trans, gapcost


def align_sequences(seqs):
    """Align the A/C strings `seqs` (given in ASCII leaf order) and return
    the list of aligned rows (strings over "AC-") minimizing the total
    pairwise cost, then the column count, then the column-major expansion.
    """
    n = len(seqs)
    lens = tuple(len(s) for s in seqs)
    preorders, init_ci, trans, gapcost = _preorder_tables(n)
    npre = len(preorders)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    masks = list(range(1, 1 << n))
    mask_bits = {m: [i for i in range(n) if (m >> i) & 1] for m in masks}
    mask_pairs = {m: [(i, j) for (i, j) in pairs
                      if (m >> i) & 1 and (m >> j) & 1] for m in masks}
    # Mixed-radix encoding of the position vector.
    base = [1] * n
    for i in range(1, n):
        base[i] = base[i - 1] * (lens[i - 1] + 1)
    mask_delta = {m: sum(base[i] for i in mask_bits[m]) for m in masks}

    # Backward pass over the DAG: dist[state] = best (cost, columns) to
    # the end.  Moves strictly increase sum(pos), so decreasing-sum order
    # is a valid reverse topological order.
    dist = [None] * (base[-1] * (lens[-1] + 1) * npre)
    positions = list(itertools.product(*(range(L + 1) for L in lens)))
    positions.sort(key=sum, reverse=True)
    full = tuple(lens)
    for pos in positions:
        pcode = 0
        for i in range(n):
            pcode += pos[i] * base[i]
        off = pcode * npre
        if pos == full:
            for ci in range(npre):
                dist[off + ci] = (0, 0)
            continue
        avail = [m for m in masks
                 if all(pos[i] < lens[i] for i in mask_bits[m])]
        nxt = [seqs[i][pos[i]] if pos[i] < lens[i] else ""
               for i in range(n)]
        for ci in range(npre):
            trow = trans[ci]
            grow = gapcost[ci]
            best = None
            for m in avail:
                mc = grow[m]
                for i, j in mask_pairs[m]:
                    if nxt[i] != nxt[j]:
                        mc += 1
                dcost, dcols = dist[(pcode + mask_delta[m]) * npre
                                    + trow[m]]
                cand = (mc + dcost, 1 + dcols)
                if best is None or cand < best:
                    best = cand
            dist[off + ci] = best

    # Forward pass: among the moves that can still attain the optimum,
    # always take the lexicographically smallest column expansion; the
    # expansion determines the path, so this yields the smallest full
    # expansion string.
    opt = dist[init_ci]
    rows = [[] for _ in range(n)]
    pos = (0,) * n
    pcode = 0
    ci = init_ci
    acc_cost = 0
    acc_cols = 0
    while pos != full:
        avail = [m for m in masks
                 if all(pos[i] < lens[i] for i in mask_bits[m])]
        nxt = [seqs[i][pos[i]] if pos[i] < lens[i] else ""
               for i in range(n)]
        best_exp = None
        best_move = None
        for m in avail:
            mc = gapcost[ci][m]
            for i, j in mask_pairs[m]:
                if nxt[i] != nxt[j]:
                    mc += 1
            ncode = pcode + mask_delta[m]
            nci = trans[ci][m]
            dcost, dcols = dist[ncode * npre + nci]
            if (acc_cost + mc + dcost, acc_cols + 1 + dcols) != opt:
                continue
            exp = "".join(nxt[i] if (m >> i) & 1 else "-"
                          for i in range(n))
            if best_exp is None or exp < best_exp:
                best_exp = exp
                best_move = (m, mc, ncode, nci)
        m, mc, pcode, ci = best_move
        for i in range(n):
            rows[i].append(nxt[i] if (m >> i) & 1 else "-")
        acc_cost += mc
        acc_cols += 1
        pos = tuple(pos[i] + ((m >> i) & 1) for i in range(n))
    return ["".join(row) for row in rows]


def run_sequences_entry(model):
    """Alignment import entry: align the leaf sequences, then run the
    original site inference on every column with gaps converted to "?"."""
    leaves = model["leaves"]
    seqs = [model["sequences"][leaf] for leaf in leaves]
    rows = align_sequences(seqs)
    alignment = {leaf: row for leaf, row in zip(leaves, rows)}
    sites = []
    for k in range(len(rows[0])):
        site = {}
        for leaf, row in zip(leaves, rows):
            site[leaf] = "?" if row[k] == "-" else row[k]
        sites.append(site)
    result = solve(model["edges"], model["root"], leaves, model["pi"],
                   model["a_q"], model["b_q"], model["rates"], sites)
    return {"alignment": alignment, "result": result}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def report_error(path, message, index=None):
    payload = {"status": "error", "path": path, "index": index,
               "message": message}
    sys.stdout.write(json.dumps(payload) + "\n")


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: python3 phylo_backend.py "
                         "INPUT.json OUTPUT.json\n")
        return 2
    input_path, output_path = argv[1], argv[2]
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        report_error("$", "cannot read input file: %s" % exc)
        return 1
    try:
        mode, model = parse_input(text)
        if mode == "sequences":
            result = run_sequences_entry(model)
        else:
            result = solve(**model)
    except Reject as rj:
        report_error(rj.path, rj.message, rj.index)
        return 1
    # Normal result: replace the output atomically.  On rejection above we
    # returned early, so a pre-existing output file is preserved.
    tmp_path = "%s.tmp.%d" % (output_path, os.getpid())
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")
        os.replace(tmp_path, output_path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        report_error("$", "cannot write output file: %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

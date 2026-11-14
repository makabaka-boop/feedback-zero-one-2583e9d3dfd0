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

On any invalid input (including duplicate or unknown members) the whole run
is rejected: an error object {"status": "error", "path": <JSON path>,
"index": <zero-based index or null>, "message": ...} is printed to stdout,
the exit code is 1 and a pre-existing OUTPUT.json is kept untouched.
JSON syntax errors are reported with path "$".
"""

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


def parse_input(text):
    def bad_constant(x):
        raise ValueError("invalid constant %s" % x)

    try:
        data = json.loads(text, object_pairs_hook=Pairs,
                          parse_constant=bad_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        reject("$", "invalid JSON syntax (%s)" % exc)
    top = _as_object(data, "$")
    required = ("edges", "pi", "Q", "rates", "sites")
    for key in top:
        if key not in required:
            reject("$." + key, "unknown top-level member %r" % (key,))
    for key in required:
        if key not in top:
            reject("$", "missing top-level member %r" % (key,))
    edges, root, leaves = check_edges(top["edges"])
    pi = check_pi(top["pi"])
    a_q, b_q = check_Q(top["Q"])
    rates = check_rates(top["rates"])
    sites = check_sites(top["sites"], set(leaves))
    return {"edges": edges, "root": root, "leaves": leaves, "pi": pi,
            "a_q": a_q, "b_q": b_q, "rates": rates, "sites": sites}


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
        model = parse_input(text)
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

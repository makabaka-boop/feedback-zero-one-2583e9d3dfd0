#!/usr/bin/env python3
"""Independent acceptance tests for phylo_backend.py.

Small trees only, with expectations derived independently of the backend:
  * closed-form coth(1) values for the worked example and variants,
  * explicit enumeration over all node-state assignments on a 3-leaf tree,
    with per-edge expected jump counts from Simpson-rule numerical
    integration of the endpoint-conditioned jump intensity,
  * rejection / output-file-preservation checks,
  * for the alignment import (sequences) entry: independent enumeration of
    all short-string alignments (joint 3/4-leaf optima, ties, empty
    strings, a double-gap-column case that discriminates the
    run-not-breaking cost rule), with per-column inference results checked
    against both the sites entry and brute-force enumeration.

Run:  python3 acceptance_test.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "phylo_backend.py")

COTH1 = (math.exp(2.0) + 1.0) / (math.exp(2.0) - 1.0)  # 1.3130352854993315

FAILURES = []


def check(name, ok, detail=""):
    print("%s %s%s" % ("ok  " if ok else "FAIL", name,
                       ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def run_backend(payload, workdir, sentinel=None):
    """Run the CLI on payload (dict or raw string); return (proc, out_text)."""
    in_path = os.path.join(workdir, "in.json")
    out_path = os.path.join(workdir, "out.json")
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    with open(in_path, "w") as fh:
        fh.write(raw)
    if sentinel is not None:
        with open(out_path, "w") as fh:
            fh.write(sentinel)
    proc = subprocess.run([sys.executable, BACKEND, in_path, out_path],
                          capture_output=True, text=True)
    out_text = None
    if os.path.exists(out_path):
        with open(out_path) as fh:
            out_text = fh.read()
    return proc, out_text


def example_payload(sites, edges=None, pi=None, rates=None, q=None):
    return {
        "edges": edges if edges is not None else [["R", "a", "0/1"],
                                                  ["R", "b", "1/1"]],
        "pi": pi if pi is not None else ["1/2", "1/2"],
        "Q": q if q is not None else [["-1/1", "1/1"], ["1/1", "-1/1"]],
        "rates": rates if rates is not None else [["1/1", "1/1"]],
        "sites": sites,
    }


# ---------------------------------------------------------------------------
# Independent reference machinery (no code shared with the backend)
# ---------------------------------------------------------------------------

def ref_P(a, b, t):
    """exp(t*Q) for Q = [[-a, a], [b, -b]], textbook closed form."""
    s = a + b
    if s == 0.0 or t == 0.0:
        return ((1.0, 0.0), (0.0, 1.0))
    u = math.exp(-s * t)
    return (((b + a * u) / s, a * (1.0 - u) / s),
            (b * (1.0 - u) / s, (a + b * u) / s))


def ref_expected_jumps(a, b, t, i, j, n=4096):
    """E[#jumps | X_0=i, X_t=j] via Simpson integration of
    sum_{k != l} P_ik(s) Q_kl P_lj(t-s) over s in [0, t], divided by P_ij(t)."""
    if t == 0.0 or a + b == 0.0:
        return 0.0
    Q = ((-a, a), (b, -b))
    pij = ref_P(a, b, t)[i][j]
    if pij == 0.0:
        return 0.0

    def integrand(s):
        P1 = ref_P(a, b, s)
        P2 = ref_P(a, b, t - s)
        tot = 0.0
        for k in (0, 1):
            for l in (0, 1):
                if l != k:
                    tot += P1[i][k] * Q[k][l] * P2[l][j]
        return tot

    h = t / n
    acc = integrand(0.0) + integrand(t)
    for m in range(1, n):
        acc += (4.0 if m % 2 else 2.0) * integrand(m * h)
    return acc * h / 3.0 / pij


def brute_force_raw(nodes, root, edges, leaves, pi, a_q, b_q, rates, site):
    """(P(E and V), P(V), numerator counts) by enumerating all
    2^len(nodes) assignments; counts are not yet divided by P(E and V)."""
    idx = {v: k for k, v in enumerate(nodes)}
    p_ev = 0.0
    p_v = 0.0
    num = [0.0] * len(edges)
    for r, w in rates:
        Ps = [ref_P(r * a_q, r * b_q, t) for (_p, _c, t) in edges]
        EN = [tuple(tuple(ref_expected_jumps(r * a_q, r * b_q, t, i, j)
                          for j in (0, 1)) for i in (0, 1))
              for (_p, _c, t) in edges]
        for code in range(1 << len(nodes)):
            st = [(code >> k) & 1 for k in range(len(nodes))]
            prob = pi[st[idx[root]]]
            for ei, (p, c, _t) in enumerate(edges):
                prob *= Ps[ei][st[idx[p]]][st[idx[c]]]
            if prob == 0.0:
                continue
            ok = True
            for leaf, obs in site.items():
                if obs != "?" and st[idx[leaf]] != "AC".index(obs):
                    ok = False
                    break
            leaf_states = [st[idx[l]] for l in leaves]
            mixed = any(x != leaf_states[0] for x in leaf_states)
            if mixed:
                p_v += w * prob
            if ok and mixed:
                p_ev += w * prob
                for ei, (p, c, _t) in enumerate(edges):
                    num[ei] += w * prob * EN[ei][st[idx[p]]][st[idx[c]]]
    return p_ev, p_v, num


def brute_force(nodes, root, edges, leaves, pi, a_q, b_q, rates, site):
    """(likelihood, counts) by enumerating all 2^len(nodes) assignments."""
    p_ev, p_v, num = brute_force_raw(nodes, root, edges, leaves, pi,
                                     a_q, b_q, rates, site)
    return p_ev / p_v, [x / p_ev for x in num]


# ---------------------------------------------------------------------------
# Independent alignment reference (enumeration; no code shared with the
# backend's alignment core)
# ---------------------------------------------------------------------------

def ref_alignments(seqs):
    """Generate every valid alignment of seqs: equal-length rows over
    "AC-", no all-gap column, gap removal restoring each original string."""
    n = len(seqs)
    rows = [[] for _ in range(n)]
    pos = [0] * n

    def rec():
        if all(pos[i] == len(seqs[i]) for i in range(n)):
            yield tuple("".join(r) for r in rows)
            return
        avail_mask = 0
        for i in range(n):
            if pos[i] < len(seqs[i]):
                avail_mask |= 1 << i
        m = avail_mask
        while m:
            for i in range(n):
                if (m >> i) & 1:
                    rows[i].append(seqs[i][pos[i]])
                    pos[i] += 1
                else:
                    rows[i].append("-")
            yield from rec()
            for i in range(n):
                rows[i].pop()
                if (m >> i) & 1:
                    pos[i] -= 1
            m = (m - 1) & avail_mask

    yield from rec()


def ref_pair_cost(x, y, breaking=False):
    """Pairwise row cost: delete double-gap columns, then identical letters
    0, different letters 1, and a maximal run of k same-side gaps k + 1
    (switching sides restarts a run).  With breaking=True, double-gap
    columns additionally cut runs -- used only to show that the
    double-gap test case discriminates the two semantics."""
    cost = 0
    run = 0
    side = None
    for a, b in zip(x, y):
        if a == "-" and b == "-":
            if breaking and run:
                cost += run + 1
                run = 0
                side = None
            continue
        if a != "-" and b != "-":
            if run:
                cost += run + 1
                run = 0
                side = None
            if a != b:
                cost += 1
            continue
        s = 0 if a == "-" else 1
        if s == side:
            run += 1
        else:
            if run:
                cost += run + 1
            run = 1
            side = s
    if run:
        cost += run + 1
    return cost


def ref_expansion(rows):
    """Column-major expansion of the alignment (rows in ASCII leaf order);
    Python's string order gives "-" < "A" < "C" as required."""
    return "".join("".join(row[k] for row in rows)
                   for k in range(len(rows[0])))


def ref_best_alignment(seqs, breaking=False):
    """The (cost, columns, expansion)-minimal alignment by enumeration."""
    best = None
    best_rows = None
    for rows in ref_alignments(seqs):
        cost = sum(ref_pair_cost(rows[i], rows[j], breaking)
                   for i in range(len(rows))
                   for j in range(i + 1, len(rows)))
        key = (cost, len(rows[0]), ref_expansion(rows))
        if best is None or key < best:
            best = key
            best_rows = rows
    return best_rows


def ref_has_spanning_run(rows):
    """True if some pair has a same-side gap run interrupted (in the raw
    alignment) by a double-gap column."""
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            rel = []
            for k in range(len(rows[0])):
                a, b = rows[i][k], rows[j][k]
                if a == "-" and b == "-":
                    rel.append("D")
                elif a == "-":
                    rel.append("i")
                elif b == "-":
                    rel.append("j")
                else:
                    rel.append("M")
            for s in range(len(rel)):
                if rel[s] not in ("i", "j"):
                    continue
                for t in range(s + 2, len(rel)):
                    mid = rel[s + 1:t]
                    if ("D" in mid and rel[t] == rel[s]
                            and all(x in (rel[s], "D") for x in mid)):
                        return True
    return False


def ref_sites_from_alignment(alignment):
    """Per-column sites with gaps converted to "?" (all columns kept)."""
    leaves = sorted(alignment)
    rows = [alignment[l] for l in leaves]
    return [{l: ("?" if r[k] == "-" else r[k])
             for l, r in zip(leaves, rows)}
            for k in range(len(rows[0]))]


def sequences_payload(seq_map, edges=None, pi=None, rates=None, q=None):
    payload = example_payload([], edges=edges, pi=pi, rates=rates, q=q)
    del payload["sites"]
    payload["sequences"] = seq_map
    return payload


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_jump_matrix_vs_quadrature():
    """Direct unit check of the backend's closed-form J against numerical
    integration, including asymmetric Q (a != b)."""
    sys.path.insert(0, HERE)
    import phylo_backend as backend
    ok = True
    detail = ""
    for a, b, t in ((1.0, 1.0, 1.0), (3.0, 2.0, 0.9), (1.5, 1.0, 0.4),
                    (0.0, 2.0, 1.3), (2.0, 0.0, 0.7), (4.0, 0.25, 2.5),
                    (0.3, 0.7, 0.01)):
        P, J = backend.transition_and_jumps(a, b, t)
        for i in (0, 1):
            for j in (0, 1):
                want = ref_expected_jumps(a, b, t, i, j)
                got = J[i][j] / P[i][j] if P[i][j] else 0.0
                if abs(got - want) > 1e-6:
                    ok = False
                    detail = ("a=%r b=%r t=%r %d->%d: %r vs %r"
                              % (a, b, t, i, j, got, want))
    check("jump_matrix: closed form matches quadrature (incl. asymmetric)",
          ok, detail)


def test_worked_example():
    with tempfile.TemporaryDirectory() as d:
        proc, out = run_backend(example_payload([{"a": "A", "b": "C"}]), d)
        check("worked_example: exit 0", proc.returncode == 0, proc.stderr)
        res = json.loads(out)
        ok = (isinstance(res, list) and len(res) == 1
              and abs(res[0]["likelihood"] - 0.5) <= 1e-8
              and len(res[0]["counts"]) == 2
              and res[0]["counts"][0] == 0.0
              and abs(res[0]["counts"][1] - COTH1) <= 1e-8)
        check("worked_example: likelihood=1/2, counts=[0, coth(1)]", ok, out)


def test_edge_order():
    with tempfile.TemporaryDirectory() as d:
        payload = example_payload([{"a": "A", "b": "C"}],
                                  edges=[["R", "b", "1/1"], ["R", "a", "0/1"]])
        proc, out = run_backend(payload, d)
        res = json.loads(out)
        ok = (proc.returncode == 0
              and abs(res[0]["counts"][0] - COTH1) <= 1e-8
              and res[0]["counts"][1] == 0.0)
        check("edge_order: counts follow input edge order", ok, out)


def test_all_missing():
    with tempfile.TemporaryDirectory() as d:
        proc, out = run_backend(example_payload([{"a": "?", "b": "?"}]), d)
        res = json.loads(out)
        ok = (proc.returncode == 0
              and abs(res[0]["likelihood"] - 1.0) <= 1e-8
              and res[0]["counts"][0] == 0.0
              and abs(res[0]["counts"][1] - COTH1) <= 1e-8)
        check("all_missing: likelihood=1, counts=[0, coth(1)]", ok, out)


def test_impossible_constant():
    with tempfile.TemporaryDirectory() as d:
        proc, out = run_backend(example_payload([{"a": "A", "b": "A"}]), d)
        res = json.loads(out)
        check("impossible_constant: only {status: impossible}",
              proc.returncode == 0 and res == [{"status": "impossible"}], out)


def test_unobservable():
    cases = {
        "zero rate": example_payload([{"a": "A", "b": "C"}],
                                     rates=[["0/1", "1/1"]]),
        "zero Q": example_payload([{"a": "A", "b": "C"}],
                                  q=[["0/1", "0/1"], ["0/1", "0/1"]]),
        "zero lengths": example_payload([{"a": "A", "b": "C"}],
                                        edges=[["R", "a", "0/1"],
                                               ["R", "b", "0/1"]]),
    }
    for name, payload in cases.items():
        with tempfile.TemporaryDirectory() as d:
            proc, out = run_backend(payload, d)
            check("unobservable (%s)" % name,
                  proc.returncode == 0 and json.loads(out) ==
                  {"status": "unobservable"}, out)


def test_empty_sites_precedence():
    with tempfile.TemporaryDirectory() as d:
        # Empty sites wins even when P(V) = 0 (zero rate).
        payload = example_payload([], rates=[["0/1", "1/1"]])
        proc, out = run_backend(payload, d)
        check("empty_sites: [] takes precedence over unobservable",
              proc.returncode == 0 and json.loads(out) == [], out)


def test_rate_mixture():
    with tempfile.TemporaryDirectory() as d:
        payload = example_payload([{"a": "A", "b": "C"}],
                                  rates=[["0/1", "1/2"], ["1/1", "1/2"]])
        proc, out = run_backend(payload, d)
        res = json.loads(out)
        ok = (proc.returncode == 0
              and abs(res[0]["likelihood"] - 0.5) <= 1e-8
              and res[0]["counts"][0] == 0.0
              and abs(res[0]["counts"][1] - COTH1) <= 1e-8)
        check("rate_mixture: mix joint probabilities, then condition", ok, out)


def test_deterministic_root():
    with tempfile.TemporaryDirectory() as d:
        payload = example_payload([{"a": "A", "b": "?"}], pi=["1/1", "0/1"])
        proc, out = run_backend(payload, d)
        res = json.loads(out)
        ok = (proc.returncode == 0
              and abs(res[0]["likelihood"] - 1.0) <= 1e-8
              and abs(res[0]["counts"][1] - COTH1) <= 1e-8)
        check("deterministic_root: compatible site", ok, out)
    with tempfile.TemporaryDirectory() as d:
        payload = example_payload([{"a": "C", "b": "?"}], pi=["1/1", "0/1"])
        proc, out = run_backend(payload, d)
        check("deterministic_root: incompatible site",
              proc.returncode == 0 and json.loads(out) ==
              [{"status": "impossible"}], out)


def test_brute_force_three_leaves():
    nodes = ["R", "u", "a", "b", "c"]
    root = "R"
    edges_f = [("R", "u", 0.4), ("R", "b", 0.7), ("u", "a", 0.2),
               ("u", "c", 0.9)]
    leaves = ["a", "b", "c"]
    pi = [0.4, 0.6]
    a_q, b_q = 3.0, 2.0
    rates = [(0.5, 1.0 / 3.0), (1.5, 2.0 / 3.0)]
    sites = [{"a": "A", "b": "?", "c": "C"},
             {"a": "?", "b": "?", "c": "?"},
             {"a": "A", "b": "A", "c": "?"}]
    payload = {
        "edges": [["R", "u", "2/5"], ["R", "b", "7/10"],
                  ["u", "a", "1/5"], ["u", "c", "9/10"]],
        "pi": ["2/5", "3/5"],
        "Q": [["-3/1", "3/1"], ["2/1", "-2/1"]],
        "rates": [["1/2", "1/3"], ["3/2", "2/3"]],
        "sites": sites,
    }
    with tempfile.TemporaryDirectory() as d:
        proc, out = run_backend(payload, d)
        if proc.returncode != 0:
            check("brute_force_3leaves: exit 0", False, proc.stdout)
            return
        res = json.loads(out)
        ok_all = True
        detail = ""
        for k, site in enumerate(sites):
            like, counts = brute_force(nodes, root, edges_f, leaves, pi,
                                       a_q, b_q, rates, site)
            got = res[k]
            if abs(got["likelihood"] - like) > 1e-8:
                ok_all = False
                detail = "site %d likelihood %r vs %r" % (k,
                                                          got["likelihood"],
                                                          like)
            for e in range(len(edges_f)):
                if abs(got["counts"][e] - counts[e]) > 1e-6:
                    ok_all = False
                    detail = ("site %d edge %d count %r vs %r"
                              % (k, e, got["counts"][e], counts[e]))
        check("brute_force_3leaves: likelihood & counts match enumeration "
              "+ quadrature", ok_all, detail or out)


def test_site_order():
    with tempfile.TemporaryDirectory() as d:
        sites = [{"a": "A", "b": "C"}, {"a": "?", "b": "?"},
                 {"a": "A", "b": "A"}]
        proc, out = run_backend(example_payload(sites), d)
        res = json.loads(out)
        ok = (proc.returncode == 0 and len(res) == 3
              and abs(res[0]["likelihood"] - 0.5) <= 1e-8
              and abs(res[1]["likelihood"] - 1.0) <= 1e-8
              and res[2] == {"status": "impossible"})
        check("site_order: results in input site order", ok, out)


def test_rejections():
    valid = example_payload([{"a": "A", "b": "C"}])

    def mutated(**kw):
        m = example_payload([{"a": "A", "b": "C"}])
        m.update(kw)
        return m

    cases = [
        ("syntax error", '{"edges": [', "$"),
        ("not an object", '[1, 2, 3]', None),
        ("duplicate top member",
         '{"edges": [], "edges": [], "pi": [], "Q": [], "rates": [],'
         ' "sites": []}', None),
        ("unknown top member", mutated(extra=1), None),
        ("missing member", {k: v for k, v in valid.items() if k != "rates"},
         None),
        ("edges not array", mutated(edges={}), None),
        ("edge not triple", mutated(edges=[["R", "a"]]), None),
        ("bad identifier", mutated(edges=[["R", "a_1", "1/1"],
                                          ["R", "b", "1/1"]]), None),
        ("empty identifier", mutated(edges=[["", "a", "1/1"],
                                            ["R", "b", "1/1"]]), None),
        ("self loop", mutated(edges=[["R", "R", "1/1"], ["R", "b", "1/1"]]),
         None),
        ("negative t", mutated(edges=[["R", "a", "-1/2"], ["R", "b", "1/1"]]),
         None),
        ("t not p/q string", mutated(edges=[["R", "a", 0.5],
                                            ["R", "b", "1/1"]]), None),
        ("zero denominator", mutated(edges=[["R", "a", "1/0"],
                                            ["R", "b", "1/1"]]), None),
        ("two roots", mutated(edges=[["R", "a", "1/1"], ["S", "b", "1/1"]]),
         None),
        ("duplicate child", mutated(edges=[["R", "a", "1/1"],
                                           ["S", "a", "1/1"]]), None),
        ("cycle no root", mutated(edges=[["R", "a", "1/1"],
                                         ["a", "R", "1/1"]]), None),
        ("single leaf", mutated(edges=[["R", "a", "1/1"]]), None),
        ("empty edges", mutated(edges=[]), None),
        ("pi wrong length", mutated(pi=["1/2"]), None),
        ("pi not sum 1", mutated(pi=["1/3", "1/3"]), None),
        ("pi negative", mutated(pi=["-1/2", "3/2"]), None),
        ("Q not 2x2", mutated(Q=[["-1/1", "1/1"]]), None),
        ("Q off-diagonal negative",
         mutated(Q=[["-1/1", "1/1"], ["-2/1", "2/1"]]), None),
        ("Q row sum nonzero",
         mutated(Q=[["-1/1", "2/1"], ["1/1", "-1/1"]]), None),
        ("rates empty", mutated(rates=[]), None),
        ("rate weight zero", mutated(rates=[["1/1", "0/1"]]), None),
        ("rate weights not sum 1", mutated(rates=[["1/1", "1/2"]]), None),
        ("rate negative", mutated(rates=[["-1/1", "1/1"]]), None),
        ("sites not array", mutated(sites={}), None),
        ("site missing leaf", mutated(sites=[{"a": "A"}]), None),
        ("site unknown leaf", mutated(sites=[{"a": "A", "b": "C",
                                              "z": "A"}]), None),
        ("site internal node", mutated(sites=[{"a": "A", "R": "C"}]), None),
        ("site bad value", mutated(sites=[{"a": "A", "b": "B"}]), None),
        ("site duplicate member",
         '{"edges": [["R","a","0/1"],["R","b","1/1"]],'
         ' "pi": ["1/2","1/2"],'
         ' "Q": [["-1/1","1/1"],["1/1","-1/1"]],'
         ' "rates": [["1/1","1/1"]],'
         ' "sites": [{"a": "A", "a": "C", "b": "C"}]}', None),
    ]
    for name, payload, want_path in cases:
        with tempfile.TemporaryDirectory() as d:
            proc, out = run_backend(payload, d, sentinel="SENTINEL")
            ok = proc.returncode == 1 and out == "SENTINEL"
            detail = ""
            if ok:
                try:
                    err = json.loads(proc.stdout)
                    ok = (err.get("status") == "error"
                          and isinstance(err.get("path"), str)
                          and "index" in err
                          and isinstance(err.get("message"), str))
                    if want_path is not None:
                        ok = ok and err["path"] == want_path
                    detail = proc.stdout.strip()
                except (json.JSONDecodeError, AttributeError):
                    ok = False
                    detail = "stdout is not an error JSON: %r" % proc.stdout
            check("reject: %s" % name, ok, detail)


def test_output_replaced_on_success():
    with tempfile.TemporaryDirectory() as d:
        proc1, out1 = run_backend(example_payload([{"a": "A", "b": "C"}]), d,
                                  sentinel="SENTINEL")
        ok1 = proc1.returncode == 0 and out1 != "SENTINEL"
        first = out1
        proc2, out2 = run_backend(example_payload([{"a": "?", "b": "?"}]), d)
        ok2 = (proc2.returncode == 0 and out2 != first
               and abs(json.loads(out2)[0]["likelihood"] - 1.0) <= 1e-8)
        check("output replaced on success, kept on rejection", ok1 and ok2,
              "%r -> %r" % (first, out2))


# ---------------------------------------------------------------------------
# Alignment import entry (sequences) tests
# ---------------------------------------------------------------------------

# Reference models: (nodes, root, edges_float, leaves, pi, a_q, b_q, rates)
REF2 = (["R", "a", "b"], "R", [("R", "a", 0.0), ("R", "b", 1.0)],
        ["a", "b"], [0.5, 0.5], 1.0, 1.0, [(1.0, 1.0)])

EDGES3 = [["R", "u", "2/5"], ["R", "b", "7/10"], ["u", "a", "1/5"],
          ["u", "c", "9/10"]]
PI3 = ["2/5", "3/5"]
Q3 = [["-3/1", "3/1"], ["2/1", "-2/1"]]
RATES3 = [["1/2", "1/3"], ["3/2", "2/3"]]
REF3 = (["R", "u", "a", "b", "c"], "R",
        [("R", "u", 0.4), ("R", "b", 0.7), ("u", "a", 0.2), ("u", "c", 0.9)],
        ["a", "b", "c"], [0.4, 0.6], 3.0, 2.0,
        [(0.5, 1.0 / 3.0), (1.5, 2.0 / 3.0)])

EDGES4 = [["R", "u", "2/5"], ["R", "v", "3/10"], ["u", "a", "1/5"],
          ["u", "b", "7/10"], ["v", "c", "1/2"], ["v", "d", "4/5"]]
REF4 = (["R", "u", "v", "a", "b", "c", "d"], "R",
        [("R", "u", 0.4), ("R", "v", 0.3), ("u", "a", 0.2), ("u", "b", 0.7),
         ("v", "c", 0.5), ("v", "d", 0.8)],
        ["a", "b", "c", "d"], [0.4, 0.6], 3.0, 2.0,
        [(0.5, 1.0 / 3.0), (1.5, 2.0 / 3.0)])


def check_columns_brute(name, result, sites, ref):
    """Verify each column's inference result against brute-force
    enumeration over all node-state assignments."""
    nodes, root, edges_f, leaves, pi_f, a_q, b_q, rates_f = ref
    if not isinstance(result, list) or len(result) != len(sites):
        return False, "result %r is not a %d-column list" % (result,
                                                             len(sites))
    for k, (entry, site) in enumerate(zip(result, sites)):
        p_ev, p_v, num = brute_force_raw(nodes, root, edges_f, leaves,
                                         pi_f, a_q, b_q, rates_f, site)
        if p_ev == 0.0:
            if entry != {"status": "impossible"}:
                return False, "column %d: %r != impossible" % (k, entry)
            continue
        if not isinstance(entry, dict) or "likelihood" not in entry \
                or "counts" not in entry:
            return False, "column %d: malformed entry %r" % (k, entry)
        if abs(entry["likelihood"] - p_ev / p_v) > 1e-8:
            return False, "column %d likelihood %r vs %r" % (
                k, entry["likelihood"], p_ev / p_v)
        for e in range(len(edges_f)):
            if abs(entry["counts"][e] - num[e] / p_ev) > 1e-6:
                return False, "column %d edge %d count %r vs %r" % (
                    k, e, entry["counts"][e], num[e] / p_ev)
    return True, ""


def run_align_case(name, payload, ref, expect=None):
    """Run the sequences entry: the alignment must match independent
    enumeration (and `expect` if given), and the result must match both
    the sites entry on the derived columns and brute-force inference."""
    with tempfile.TemporaryDirectory() as d:
        proc, out = run_backend(payload, d)
        if proc.returncode != 0:
            check(name, False, "exit %d: %s" % (proc.returncode,
                                                proc.stdout))
            return
        res = json.loads(out)
        seq_map = payload["sequences"]
        leaves_sorted = sorted(seq_map)
        ok = set(res) == {"alignment", "result"}
        detail = out
        sites = None
        if ok:
            want_rows = ref_best_alignment([seq_map[l]
                                            for l in leaves_sorted])
            want = {l: r for l, r in zip(leaves_sorted, want_rows)}
            if res["alignment"] != want:
                ok = False
                detail = ("alignment %r != enumeration %r"
                          % (res["alignment"], want))
            elif expect is not None and res["alignment"] != expect:
                ok = False
                detail = ("alignment %r != expected %r"
                          % (res["alignment"], expect))
        if ok:
            sites = ref_sites_from_alignment(res["alignment"])
            sites_payload = {k: v for k, v in payload.items()
                             if k != "sequences"}
            sites_payload["sites"] = sites
            proc2, out2 = run_backend(sites_payload, d)
            ok = (proc2.returncode == 0
                  and res["result"] == json.loads(out2))
            detail = ("result %r != sites-entry result %r"
                      % (res["result"], out2))
        if ok:
            ok, detail = check_columns_brute(name, res["result"], sites,
                                             ref)
        check(name, ok, detail)


def test_align_spec_example():
    with tempfile.TemporaryDirectory() as d:
        payload = sequences_payload({"a": "A", "b": "CA"})
        proc, out = run_backend(payload, d)
        res = json.loads(out) if proc.returncode == 0 else {}
        ok = (proc.returncode == 0
              and set(res) == {"alignment", "result"}
              and res["alignment"] == {"a": "-A", "b": "CA"})
        # Columns: {a: "?", b: "C"} then {a: "A", b: "A"}.
        sites = [{"a": "?", "b": "C"}, {"a": "A", "b": "A"}]
        proc2, out2 = run_backend(example_payload(sites), d)
        ok = (ok and proc2.returncode == 0
              and res["result"] == json.loads(out2))
        ok = (ok and len(res["result"]) == 2
              and abs(res["result"][0]["likelihood"] - 0.5) <= 1e-8
              and res["result"][0]["counts"][0] == 0.0
              and abs(res["result"][0]["counts"][1] - COTH1) <= 1e-8
              and res["result"][1] == {"status": "impossible"})
        check("align_spec_example: a=A, b=CA -> a=-A, b=CA with per-column "
              "inference", ok, out)


def test_align_two_leaf_enumeration():
    cases = [
        ("spec", {"a": "A", "b": "CA"}, {"a": "-A", "b": "CA"}),
        # Tie: ("AA", "-A") and ("AA", "A-") both cost 2 in 2 columns;
        # expansion "A-AA" < "AAA-" picks the former.
        ("tie", {"a": "AA", "b": "A"}, {"a": "AA", "b": "-A"}),
        ("tie_rev", {"a": "A", "b": "AA"}, {"a": "-A", "b": "AA"}),
        ("one_empty", {"a": "", "b": "A"}, {"a": "-", "b": "A"}),
        ("first_empty", {"a": "", "b": "CA"}, {"a": "--", "b": "CA"}),
        ("both_empty", {"a": "", "b": ""}, {"a": "", "b": ""}),
        ("mismatch", {"a": "AC", "b": "CA"}, None),
        ("shift3", {"a": "CCA", "b": "AC"}, None),
    ]
    for sub, seq_map, expect in cases:
        run_align_case("align_2leaf(%s)" % sub,
                       sequences_payload(seq_map), REF2, expect=expect)


def test_align_three_leaf_joint():
    # "joint_tie": aligning c="A" against a="AC" alone would give "A-";
    # the joint optimum (cost 7, tie broken by expansion "AC-CAA") is "-A".
    cases = [
        ("joint_tie", {"a": "AC", "b": "CA", "c": "A"},
         {"a": "AC", "b": "CA", "c": "-A"}),
        ("joint_shift", {"a": "A", "b": "CA", "c": "AC"},
         {"a": "-A", "b": "CA", "c": "AC"}),
        ("empty_mid", {"a": "AA", "b": "", "c": "CA"},
         {"a": "AA", "b": "--", "c": "CA"}),
        ("all_empty", {"a": "", "b": "", "c": ""},
         {"a": "", "b": "", "c": ""}),
    ]
    for sub, seq_map, expect in cases:
        run_align_case("align_3leaf(%s)" % sub,
                       sequences_payload(seq_map, edges=EDGES3, pi=PI3,
                                         rates=RATES3, q=Q3),
                       REF3, expect=expect)


def test_align_four_leaf_joint():
    cases = [
        ("joint", {"a": "AC", "b": "C", "c": "CA", "d": "A"},
         {"a": "AC", "b": "-C", "c": "CA", "d": "-A"}),
        ("with_empty", {"a": "AA", "b": "A", "c": "CA", "d": ""},
         {"a": "AA", "b": "-A", "c": "CA", "d": "--"}),
    ]
    for sub, seq_map, expect in cases:
        run_align_case("align_4leaf(%s)" % sub,
                       sequences_payload(seq_map, edges=EDGES4, pi=PI3,
                                         rates=RATES3, q=Q3),
                       REF4, expect=expect)


def test_align_double_gap_segment():
    # Optimum: a="---", b="A-A", c="ACA".  For pair (a, b) the relevant
    # columns are 1 and 3 (both gap-in-a): one run of length 2 costing 3,
    # spanning the double-gap column 2.  If double-gap columns cut runs,
    # the pair would cost 2+2=4 and a different alignment would win, so
    # this case discriminates the two semantics.
    seq_map = {"a": "", "b": "AA", "c": "ACA"}
    seqs = [seq_map[l] for l in sorted(seq_map)]
    rows = ref_best_alignment(seqs)
    breaking_rows = ref_best_alignment(seqs, breaking=True)
    ok = (rows == ("---", "A-A", "ACA")
          and ref_has_spanning_run(rows)
          and breaking_rows != rows)
    check("align_double_gap: reference case spans a double-gap column and "
          "discriminates run-breaking costs", ok,
          "rows=%r breaking=%r" % (rows, breaking_rows))
    run_align_case("align_double_gap: backend matches non-breaking "
                   "enumeration",
                   sequences_payload(seq_map, edges=EDGES3, pi=PI3,
                                     rates=RATES3, q=Q3),
                   REF3, expect={"a": "---", "b": "A-A", "c": "ACA"})


def test_align_empty_and_unobservable():
    with tempfile.TemporaryDirectory() as d:
        payload = sequences_payload({"a": "", "b": ""})
        proc, out = run_backend(payload, d)
        ok = (proc.returncode == 0
              and json.loads(out) == {"alignment": {"a": "", "b": ""},
                                      "result": []})
        check("align_empty: all-empty sequences -> empty rows, result []",
              ok, out)
    with tempfile.TemporaryDirectory() as d:
        # Empty alignment (result []) takes precedence over unobservable.
        payload = sequences_payload({"a": "", "b": ""},
                                    rates=[["0/1", "1/1"]])
        proc, out = run_backend(payload, d)
        ok = (proc.returncode == 0
              and json.loads(out)["result"] == [])
        check("align_empty: [] takes precedence over unobservable", ok, out)
    with tempfile.TemporaryDirectory() as d:
        payload = sequences_payload({"a": "A", "b": "C"},
                                    rates=[["0/1", "1/1"]])
        proc, out = run_backend(payload, d)
        res = json.loads(out) if proc.returncode == 0 else {}
        ok = (proc.returncode == 0
              and res == {"alignment": {"a": "A", "b": "C"},
                          "result": {"status": "unobservable"}})
        check("align_unobservable: alignment emitted, result keeps the "
              "original type", ok, out)


def test_align_rejections():
    five_leaf_edges = [["R", "u", "1/1"], ["R", "v", "1/1"],
                       ["u", "a", "1/1"], ["u", "b", "1/1"],
                       ["v", "c", "1/1"], ["v", "d", "1/1"],
                       ["R", "e", "1/1"]]

    def mutated(**kw):
        m = sequences_payload({"a": "A", "b": "CA"})
        m.update(kw)
        return m

    no_sites = example_payload([{"a": "A", "b": "C"}])
    del no_sites["sites"]

    cases = [
        ("five leaves",
         mutated(edges=five_leaf_edges,
                 sequences={"a": "A", "b": "A", "c": "A", "d": "A",
                            "e": "A"}), "$.edges"),
        ("sequences not object", mutated(sequences=["A", "CA"]),
         "$.sequences"),
        ("sequences unknown leaf",
         mutated(sequences={"a": "A", "b": "CA", "z": "A"}), "$.sequences"),
        ("sequences missing leaf", mutated(sequences={"a": "A"}),
         "$.sequences"),
        ("sequences internal node",
         mutated(sequences={"a": "A", "R": "CA"}), "$.sequences"),
        ("sequences bad letter",
         mutated(sequences={"a": "A", "b": "B"}), "$.sequences.b"),
        ("sequences gap char",
         mutated(sequences={"a": "A", "b": "C-"}), "$.sequences.b"),
        ("sequences missing-data char",
         mutated(sequences={"a": "?", "b": "CA"}), "$.sequences.a"),
        ("sequences lowercase",
         mutated(sequences={"a": "a", "b": "CA"}), "$.sequences.a"),
        ("sequences non-string", mutated(sequences={"a": 1, "b": "CA"}),
         "$.sequences.a"),
        ("sequences null", mutated(sequences={"a": None, "b": "CA"}),
         "$.sequences.a"),
        ("sequences duplicate member",
         '{"edges": [["R","a","0/1"],["R","b","1/1"]],'
         ' "pi": ["1/2","1/2"],'
         ' "Q": [["-1/1","1/1"],["1/1","-1/1"]],'
         ' "rates": [["1/1","1/1"]],'
         ' "sequences": {"a": "A", "a": "C", "b": "CA"}}', "$.sequences"),
        ("both sites and sequences",
         mutated(sites=[{"a": "A", "b": "C"}]), "$.sequences"),
        ("neither sites nor sequences", no_sites, "$"),
    ]
    for name, payload, want_path in cases:
        with tempfile.TemporaryDirectory() as d:
            proc, out = run_backend(payload, d, sentinel="SENTINEL")
            ok = proc.returncode == 1 and out == "SENTINEL"
            detail = ""
            if ok:
                try:
                    err = json.loads(proc.stdout)
                    ok = (err.get("status") == "error"
                          and err.get("path") == want_path
                          and err.get("index") is None
                          and isinstance(err.get("message"), str))
                    detail = proc.stdout.strip()
                except (json.JSONDecodeError, AttributeError):
                    ok = False
                    detail = "stdout is not an error JSON: %r" % proc.stdout
            check("align_reject: %s" % name, ok, detail)


def test_align_output_replaced():
    with tempfile.TemporaryDirectory() as d:
        proc1, out1 = run_backend(sequences_payload({"a": "A", "b": "CA"}),
                                  d, sentinel="SENTINEL")
        ok1 = proc1.returncode == 0 and out1 != "SENTINEL"
        # An invalid sequences input keeps the previous output file.
        proc2, out2 = run_backend(sequences_payload({"a": "A", "b": "B"}),
                                  d)
        ok2 = proc2.returncode == 1 and out2 == out1
        # The original sites entry still replaces it afterwards.
        proc3, out3 = run_backend(example_payload([{"a": "A", "b": "C"}]), d)
        ok3 = proc3.returncode == 0 and out3 != out1
        check("align_output: replaced on success, kept on rejection",
              ok1 and ok2 and ok3, "%r -> %r -> %r" % (out1, out2, out3))


def main():
    tests = [
        test_jump_matrix_vs_quadrature,
        test_worked_example,
        test_edge_order,
        test_all_missing,
        test_impossible_constant,
        test_unobservable,
        test_empty_sites_precedence,
        test_rate_mixture,
        test_deterministic_root,
        test_brute_force_three_leaves,
        test_site_order,
        test_rejections,
        test_output_replaced_on_success,
        test_align_spec_example,
        test_align_two_leaf_enumeration,
        test_align_three_leaf_joint,
        test_align_four_leaf_joint,
        test_align_double_gap_segment,
        test_align_empty_and_unobservable,
        test_align_rejections,
        test_align_output_replaced,
    ]
    for t in tests:
        t()
    print("-" * 60)
    if FAILURES:
        print("%d FAILURE(S): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("all %d acceptance tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())

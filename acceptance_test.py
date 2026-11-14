#!/usr/bin/env python3
"""Independent acceptance tests for phylo_backend.py.

Small trees only, with expectations derived independently of the backend:
  * closed-form coth(1) values for the worked example and variants,
  * explicit enumeration over all node-state assignments on a 3-leaf tree,
    with per-edge expected jump counts from Simpson-rule numerical
    integration of the endpoint-conditioned jump intensity,
  * rejection / output-file-preservation checks.

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


def brute_force(nodes, root, edges, leaves, pi, a_q, b_q, rates, site):
    """(likelihood, counts) by enumerating all 2^len(nodes) assignments."""
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
    return p_ev / p_v, [x / p_ev for x in num]


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

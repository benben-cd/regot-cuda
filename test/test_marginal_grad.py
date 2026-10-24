"""Single-input marginal VJPs: dense oracle, feasible FD and autograd contracts.

Run after building/installing the candidate:
    python -B -m unittest discover -s test -p test_marginal_grad.py -v
"""

import gc
import unittest
from unittest.mock import patch

import torch
import curegot

from test_backward import dense_gradient, problem, random_gradient


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA float64")
class TestMarginalGrad(unittest.TestCase):
    modes = ("none", "diag", "splr")
    metrics = {}

    @classmethod
    def record(cls, name, value):
        cls.metrics[name] = max(cls.metrics.get(name, 0.0), float(value))

    @classmethod
    def tearDownClass(cls):
        for name, value in sorted(cls.metrics.items()):
            print(f"MARGINAL_METRIC {name}={value:.12e}", flush=True)

    @staticmethod
    def small_problem():
        M = torch.tensor([[0.0, 0.2, 0.4, 0.1],
                          [0.3, 0.0, 0.2, 0.4],
                          [0.1, 0.4, 0.0, 0.2]], device="cuda", dtype=torch.float64)
        # Weak but nonseparable costs let the default Sinkhorn candidate
        # reach FD accuracy; harder active-rank2 cases are tested separately.
        M = 0.01 * M
        a = M.new_tensor([0.25, 0.25, 0.5])
        b = M.new_tensor([0.125, 0.25, 0.125, 0.5])
        return M, a, b

    def feasible(self, T, a, b, atol=2e-10):
        self.assertTrue(bool(torch.isfinite(a).all() & (a > 0).all()))
        self.assertTrue(bool(torch.isfinite(b).all() & (b > 0).all()))
        self.assertLessEqual(abs(float(a.sum() - b.sum())), 2e-14)
        self.assertTrue(bool(torch.isfinite(T).all()))
        error = max(float((T.sum(1) - a).abs().max()),
                    float((T.sum(0) - b).abs().max()))
        self.assertLessEqual(error, atol, f"full marginal residual {error:.3e}")
        self.record("full_marginal_residual", error)

    def plan(self, M, a, b, reg, mode="diag", tol=1e-11, feasibility_tol=2e-10):
        T = curegot.torch.sinkhorn_autograd(
            M, a, b, reg, tol=tol, max_iter=1000,
            backward_preconditioner=mode, backward_tol=1e-12,
        )
        self.feasible(T.detach(), a.detach(), b.detach(), feasibility_tol)
        return T

    @staticmethod
    def dense_oracle(T, G, reg):
        # Independent explicit Q and direct solve, rather than production assembly.
        R = T[:, :-1]
        Q = torch.cat([torch.cat([torch.diag(T.sum(1)), R], 1),
                       torch.cat([R.T, torch.diag(T.sum(0)[:-1])], 1)], 0)
        weighted = G * T
        rhs = torch.cat([weighted.sum(1), weighted.sum(0)[:-1]])
        adjoint = torch.linalg.solve(Q, rhs)
        ga = adjoint[:T.shape[0]]
        gb = torch.cat([adjoint[T.shape[0]:], adjoint.new_zeros(1)])
        return Q, rhs, (dense_gradient(T, G, reg), ga, gb)

    def close_vjps(self, actual, expected, scale=1.0, atol=2e-9, rtol=2e-8):
        for value, reference in zip(actual, expected):
            value = value / scale
            torch.testing.assert_close(value, reference, atol=atol, rtol=rtol)
            error = torch.linalg.vector_norm(value - reference)
            denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
            self.record("dense_vjp_relative_error", error / denominator)

    def finite_difference(self, loss_at, expected, name):
        # Binary step preserves the exact masses of dyadic marginal directions.
        eps = 2.0 ** -12
        with torch.no_grad():
            observed = float((loss_at(eps) - loss_at(-eps)) / (2 * eps))
        expected = float(expected)
        error = abs(observed - expected)
        self.assertLessEqual(error, 2e-7 + 2e-5 * abs(expected),
                             f"{name}: fd={observed:.12e}, vjp={expected:.12e}")
        self.record(name + "_fd_absolute_error", error)
        self.record(name + "_fd_relative_error",
                    error / max(abs(observed), abs(expected), 1e-12))

    def test_wiring_dense_oracle_and_one_native_solve(self):
        masks = ((True, False, False), (False, True, False),
                 (False, False, True), (False, True, True), (True, True, True))
        for reg in (0.2, 0.5):
            for mode in self.modes:
                for needs in masks:
                    with self.subTest(reg=reg, mode=mode, needs=needs):
                        inputs = tuple(x.clone().requires_grad_(need)
                                       for x, need in zip(self.small_problem(), needs))
                        M, a, b = inputs
                        G = random_gradient(M)
                        T = self.plan(M, a, b, reg, mode)
                        Q, rhs, reference = self.dense_oracle(T.detach(), G, reg)
                        if all(needs):
                            result = curegot.torch.solve_backward(
                                T.detach(), G, T.grad_fn.backward_context, reg,
                                preconditioner=mode, tol=1e-12,
                            )
                            self.assertTrue(result["converged"])
                            residual = torch.linalg.vector_norm(Q @ result["lambda"] - rhs)
                            relative = residual / torch.linalg.vector_norm(rhs)
                            self.assertLess(float(relative), 2e-10)
                            self.record("true_relative_residual", relative)
                            self.close_vjps((result["grad"], result["grad_a"], result["grad_b"]), reference)
                        with patch.object(curegot.torch, "_solve_backward_native",
                                          wraps=curegot.torch._solve_backward_native) as solve:
                            T.backward(G)
                        self.assertEqual(solve.call_count, 1)
                        for value, need, expected in zip(inputs, needs, reference):
                            if need:
                                self.assertIsNotNone(value.grad)
                                self.assertEqual(value.grad.shape, value.shape)
                                self.assertEqual(value.grad.dtype, value.dtype)
                                self.assertEqual(value.grad.device, value.device)
                                self.close_vjps((value.grad,), (expected,))
                            else:
                                self.assertIsNone(value.grad)
                        if needs[2]:
                            self.assertEqual(float(b.grad[-1]), 0.0)
        # No grad requested: the public call still works without changing its API.
        T = self.plan(*self.small_problem(), reg=0.5)
        self.assertFalse(T.requires_grad)

    def test_feasible_marginal_finite_differences(self):
        for reg in (0.2, 0.5):
            for mode in self.modes:
                with self.subTest(reg=reg, mode=mode):
                    M, a, b = self.small_problem()
                    a.requires_grad_(True)
                    b.requires_grad_(True)
                    G = random_gradient(M, seed=123)
                    T = self.plan(M, a, b, reg, mode)
                    (T * G).sum().backward()
                    ga, gb = a.grad, b.grad
                    a0, b0 = a.detach(), b.detach()
                    for name, marginal, gradient in (("a", a0, ga), ("b", b0, gb)):
                        for i in range(marginal.numel() - 1):
                            direction = torch.zeros_like(marginal)
                            direction[i], direction[-1] = 1.0, -1.0
                            def loss_at(step):
                                aa = a0 + step * direction if name == "a" else a0
                                bb = b0 + step * direction if name == "b" else b0
                                return (G * self.plan(M, aa, bb, reg, mode)).sum()
                            self.finite_difference(loss_at, gradient @ direction, name)
                    # Add a direction missing from separate simplex tangents:
                    # both masses change together, including the gauge-fixed b[-1].
                    da, db = torch.zeros_like(a0), torch.zeros_like(b0)
                    da[0], db[-1] = 1.0, 1.0
                    self.finite_difference(
                        lambda h: (G * self.plan(M, a0 + h * da, b0 + h * db, reg, mode)).sum(),
                        ga @ da + gb @ db, "joint_mass",
                    )
                    self.finite_difference(
                        lambda h: (G * self.plan(M, (1 + h) * a0, (1 + h) * b0, reg, mode)).sum(),
                        ga @ a0 + gb @ b0, "mass_scaling",
                    )

    def test_softmax_chain_rule_and_trainable_shared_mass(self):
        M, _, _ = self.small_problem()
        G = random_gradient(M, seed=21)
        for mode in self.modes:
            with self.subTest(mode=mode):
                u = M.new_tensor([0.2, -0.1, 0.4], requires_grad=True)
                v = M.new_tensor([-0.2, 0.3, 0.1, 0.0], requires_grad=True)
                mass = M.new_tensor(1.25, requires_grad=True)
                p, q = u.softmax(0), v.softmax(0)
                T = self.plan(M, mass * p, mass * q, 0.5, mode)
                _, _, (_, ga, gb) = self.dense_oracle(T.detach(), G, 0.5)
                expected = (mass.detach() * p.detach() * (ga - p.detach() @ ga),
                            mass.detach() * q.detach() * (gb - q.detach() @ gb),
                            p.detach() @ ga + q.detach() @ gb)
                (T * G).sum().backward()
                self.close_vjps((u.grad, v.grad, mass.grad), expected)
                du, dv = torch.zeros_like(u), torch.zeros_like(v)
                du[0], dv[1] = 1.0, 1.0
                u0, v0, mass0 = u.detach(), v.detach(), mass.detach()
                self.finite_difference(
                    lambda h: (G * self.plan(M, mass0 * (u0 + h * du).softmax(0),
                                             mass0 * v0.softmax(0), 0.5, mode)).sum(),
                    u.grad @ du, "softmax_a",
                )
                self.finite_difference(
                    lambda h: (G * self.plan(M, mass0 * u0.softmax(0),
                                             mass0 * (v0 + h * dv).softmax(0), 0.5, mode)).sum(),
                    v.grad @ dv, "softmax_b",
                )
                self.finite_difference(
                    lambda h: (G * self.plan(M, (mass0 + h) * u0.softmax(0),
                                             (mass0 + h) * v0.softmax(0), 0.5, mode)).sum(),
                    mass.grad, "softmax_mass",
                )

    def test_zero_tiny_rhs_and_repeated_backward(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                M, a, b = (x.clone().requires_grad_(True) for x in self.small_problem())
                G = random_gradient(M)
                T = self.plan(M, a, b, 0.5, mode)
                _, _, reference = self.dense_oracle(T.detach(), G, 0.5)
                T.backward(torch.zeros_like(G), retain_graph=True)
                for value in (M, a, b):
                    self.assertEqual(int(torch.count_nonzero(value.grad)), 0)
                for scale in (1e-160, 1.0):
                    for value in (M, a, b):
                        value.grad = None
                    gc.collect()
                    T.backward(G * scale, retain_graph=True)
                    self.close_vjps((M.grad, a.grad, b.grad), reference, scale)

    def test_nondefault_stream_and_strided_upstream(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                base = self.small_problem()
                G = random_gradient(base[0])
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    M, a, b = (x.clone().requires_grad_(True) for x in base)
                    T = self.plan(M, a, b, 0.5, mode)
                    storage = torch.zeros_like(T.T.contiguous())
                    upstream = storage.T
                    self.assertFalse(upstream.is_contiguous())
                    torch.cuda._sleep(2000000)
                    storage.copy_(G.T)
                    T.backward(upstream)
                    observed = tuple(x.grad.clone() for x in (M, a, b))
                stream.synchronize()
                _, _, reference = self.dense_oracle(T.detach(), G, 0.5)
                self.close_vjps(observed, reference)

    def test_active_rank2_marginals(self):
        M, a, b = (x.requires_grad_(True) for x in problem(100, 120, 40))
        G = random_gradient(M, seed=13)
        T = self.plan(M, a, b, 0.03, "splr", tol=1e-8, feasibility_tol=2e-7)
        context = T.grad_fn.backward_context
        self.assertTrue(context.low_rank_enabled and context.adopted_forward_state)
        _, _, reference = self.dense_oracle(T.detach(), G, 0.03)
        T.backward(G)
        self.close_vjps((M.grad, a.grad, b.grad), reference, atol=2e-8, rtol=1e-7)

    def test_immediate_fallback_marginals(self):
        M, a, b = (x.requires_grad_(True) for x in problem())
        G = random_gradient(M)
        native_forward = curegot.torch.sinkhorn_splr_save_ctx
        controls = {"density": 1.0, "candidate_sinkhorn_iter": 0}
        with torch.no_grad():
            initial = curegot.torch.sinkhorn_splr(M, a, b, 0.3, 1e-8, 1000, 0, **controls)
        # Exercise the real immediate-convergence native path without adding
        # research knobs to the public differentiable signature.
        def warm_forward(*args):
            return native_forward(*args, **controls, x0=initial["dual"])
        with patch.object(curegot.torch, "sinkhorn_splr_save_ctx", side_effect=warm_forward):
            T = self.plan(M, a, b, 0.3, "splr", tol=1e-8, feasibility_tol=2e-7)
        self.assertTrue(T.grad_fn.backward_context.is_fallback)
        self.assertFalse(T.grad_fn.backward_context.low_rank_enabled)
        _, _, reference = self.dense_oracle(T.detach(), G, 0.3)
        T.backward(G)
        self.close_vjps((M.grad, a.grad, b.grad), reference)

    def test_existing_contracts_with_marginal_gradients(self):
        M, a, b = self.small_problem()
        a.requires_grad_(True)
        b.requires_grad_(True)
        with self.assertRaisesRegex(NotImplementedError, "reg"):
            curegot.torch.sinkhorn_autograd(M, a, b, M.new_tensor(0.5, requires_grad=True))
        with torch.autocast("cuda"):
            with self.assertRaisesRegex(RuntimeError, "AMP/autocast"):
                curegot.torch.sinkhorn_autograd(M, a, b, 0.5)
        T = self.plan(M, a, b, 0.5)
        G = random_gradient(M).requires_grad_(True)
        ga, gb = torch.autograd.grad(T, (a, b), G, create_graph=True)
        with self.assertRaisesRegex(RuntimeError, "once_differentiable"):
            (ga.sum() + gb.sum()).backward()
        with self.assertRaisesRegex(ValueError, "m >= 2"):
            curegot.torch.sinkhorn_autograd(M[:, :1], a, b[:1], 0.5)
        # The smallest supported shape also supports marginal differentiation.
        M = M.new_tensor([[0.1, 0.3]])
        a = M.new_tensor([1.0], requires_grad=True)
        b = M.new_tensor([0.5, 0.5], requires_grad=True)
        G = M.new_tensor([[0.7, -0.2]])
        T = self.plan(M, a, b, 0.5)
        _, _, reference = self.dense_oracle(T.detach(), G, 0.5)
        T.backward(G)
        self.close_vjps((a.grad, b.grad), reference[1:])


if __name__ == "__main__":
    unittest.main()

"""Small physical limits and differentiation checks for the full CTMRG."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

import hive_ipeps


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_neel_product_state(self):
        a = torch.tensor([[[[1., 0.]]]], dtype=torch.float64)
        result = hive_ipeps.observables([a], "3legSymmetrized1tensor", chi=1)
        self.assertAlmostEqual(result["energy_per_site"], -.375, places=12)
        self.assertAlmostEqual(result["staggered_magnetization"], .5, places=12)
        self.assertTrue(result["ctm_converged"])
        self.assertEqual(len(result["bond_correlations"]), 18)
        entropy = hive_ipeps.renyi_entropy([a], "3legSymmetrized1tensor", chi=1)
        for cut in entropy["cuts"]:
            self.assertTrue(all(abs(value) < 1e-12 for value in cut["S2"]))

    def test_independent_unrestricted6tensors_state(self):
        parameters = hive_ipeps.initial_parameters("unrestricted6tensors", 2, seed=13)
        value, environment, densities = hive_ipeps.energy(parameters, "unrestricted6tensors", chi=4)
        value.backward()
        self.assertEqual([len(sector) for sector in environment.sectors], [9, 9, 9])
        self.assertFalse(torch.allclose(environment.sectors[0][0], environment.sectors[0][1]))
        self.assertTrue(all(float(p.grad.norm()) > 1e-8 for p in parameters))
        self.assertFalse(torch.allclose(densities["AB"], densities["CD"]))
        lengths = hive_ipeps.correlation_lengths(parameters, "unrestricted6tensors", chi=4)
        self.assertEqual(len(lengths["directions"]), 9)

    def test_gradients(self):
        for ansatz in hive_ipeps.ANSATZE:
            with self.subTest(ansatz=ansatz):
                parameters = hive_ipeps.initial_parameters(ansatz, 2, seed=13)
                value = hive_ipeps.energy(parameters, ansatz, chi=4)[0]
                value.backward()
                generator = torch.Generator().manual_seed(11)
                directions = [torch.randn(p.shape, dtype=p.dtype, generator=generator)
                              for p in parameters]
                analytic = sum(float((p.grad*d).sum()) for p, d in zip(parameters, directions))
                epsilon = 1e-5
                with torch.no_grad():
                    plus = [p+epsilon*d for p, d in zip(parameters, directions)]
                    minus = [p-epsilon*d for p, d in zip(parameters, directions)]
                    numeric = float((hive_ipeps.energy(plus, ansatz, chi=4)[0] -
                                     hive_ipeps.energy(minus, ansatz, chi=4)[0]) / (2*epsilon))
                self.assertLess(abs(analytic-numeric), 2e-6 * max(1., abs(numeric)))

    def test_resume_preserves_tensor_block(self):
        from examples import optimize as command

        path = Path(__file__).resolve().parents[1] / "examples/tensor.npz"
        previous, saved_ansatz, _, _ = hive_ipeps.load_tensor(path)
        for D, ansatz in ((2, saved_ansatz), (3, saved_ansatz),
                          (2, "unrestricted6tensors")):
            with self.subTest(D=D, ansatz=ansatz), TemporaryDirectory() as output:
                arguments = ["optimize.py", "--ansatz", ansatz, "--D", str(D),
                             "--resume", str(path), "--output", output]
                with mock.patch("sys.argv", arguments), mock.patch.object(
                        command, "optimize", side_effect=RuntimeError("Initialization inspected")) as run:
                    with self.assertRaisesRegex(RuntimeError, "Initialization inspected"):
                        command.main()
                current = run.call_args.args[0]
                expected = (hive_ipeps.unit_cell(previous, saved_ansatz)
                            if ansatz == "unrestricted6tensors" else previous)
                self.assertEqual(len(current), len(expected))
                for tensor, old in zip(current, expected):
                    d = old.shape[0]
                    torch.testing.assert_close(tensor[:d, :d, :d, :], old, rtol=0, atol=0)

    def test_saved_two_tensor_state(self):
        path = Path(__file__).resolve().parents[1] / "examples/tensor.npz"
        parameters, ansatz, J1, chi = hive_ipeps.load_tensor(path)
        measured = hive_ipeps.observables(parameters, ansatz, J1, chi)
        self.assertAlmostEqual(measured["energy_per_site"], -.535373955813, places=9)
        self.assertAlmostEqual(measured["staggered_magnetization"], .369961650232, places=9)

    def test_failed_lookahead_resumes_at_chi_plus_two_D(self):
        path = Path(__file__).resolve().parents[1] / "examples/tensor.npz"
        parameters, ansatz, J1, _ = hive_ipeps.load_tensor(path)
        result = hive_ipeps.optimize(parameters, ansatz, J1, chi=1, steps=1)
        self.assertEqual([check["chi"] for check in result["chi_checks"]],
                         [[1, 3, 5], [5, 7, 9]])
        self.assertFalse(result["chi_checks"][0]["passed"])
        self.assertTrue(result["chi_checks"][1]["passed"])
        self.assertTrue(result["chi_converged"])

    def test_boundary_grows_beyond_initial_patch(self):
        path = Path(__file__).resolve().parents[1] / "examples/tensor.npz"
        parameters, ansatz, J1, _ = hive_ipeps.load_tensor(path)
        parameters = [p.requires_grad_() for p in parameters]
        value, environment, _ = hive_ipeps.energy(parameters, ansatz, J1, chi=17)
        value.backward()
        for sector in environment.sectors:
            self.assertTrue(all(t.shape == (17, 17) for t in sector[:3]))
            self.assertTrue(all(t.shape == (17, 17, 4) for t in sector[3:]))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_energy_and_gradient(self):
        for ansatz in hive_ipeps.ANSATZE:
            cpu = hive_ipeps.initial_parameters(ansatz, 2, seed=13)
            gpu = [p.detach().to("cuda").requires_grad_() for p in cpu]
            expected = hive_ipeps.energy(cpu, ansatz, chi=4)[0]
            actual = hive_ipeps.energy(gpu, ansatz, chi=4)[0]
            actual.backward()
            self.assertAlmostEqual(float(expected), float(actual), places=7)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in gpu))


if __name__ == "__main__":
    unittest.main()

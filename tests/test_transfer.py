"""Physical limits of the nine unrestricted boundary contractions."""
import unittest

import numpy as np
import opt_einsum as oe
import torch

import hive_ipeps
from hive_ipeps.tensors import contract, normalize


def cylinder_entropy(sites, cut):
    """Contract the original PEPS on a four-bond cylinder, without CTM edges."""
    sites = {name: tensor.numpy() for name, tensor in zip("abcdef", sites)}
    period = np.eye(256)
    start = (6, 4, 2)[cut]
    for column in range(start, start + 6):
        arguments = []
        for first in range(column % 2, 4, 2):
            second = (first + 1) % 4
            pair = []
            for row in (first, second):
                name = ("cfabed" if row % 2 == 0 else "bedcfa")[column % 6]
                tensor = sites[name]
                if (row + column) % 2 == 0:
                    tensor = tensor.transpose(0, 2, 1, 3)
                pair.append(tensor)
            block = np.einsum("alrp,amnq->lmrnpq", *pair)
            arguments += [block, [first, second, 4+first, 4+second, 8+first, 8+second]]
        kraus = oe.contract(*arguments, list(range(12))).reshape(16, 16, 16).transpose(2, 1, 0)
        channel = np.einsum("soi,spj->opij", kraus, kraus).reshape(256, 256)
        period = channel @ period
        period /= np.linalg.norm(period)

    def fixed_point(channel):
        state = np.eye(16) / 16
        for _ in range(500):
            next_state = (channel @ state.reshape(-1)).reshape(16, 16)
            next_state /= np.trace(next_state)
            if np.linalg.norm(next_state-state) < 1e-13:
                return next_state
            state = next_state
        raise AssertionError("The tiny-cylinder fixed point did not settle.")

    density = fixed_point(period) @ fixed_point(period.T)
    return -np.log(np.trace(density @ density) / np.trace(density)**2)


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_product_spins(self):
        angles = torch.arange(6, dtype=torch.float64) * .31
        sites = [torch.stack((a.cos(), a.sin())).reshape(1, 1, 1, 2) for a in angles]
        lengths = hive_ipeps.correlation_lengths(sites, "unrestricted6tensors", chi=1)["directions"]
        entropy = hive_ipeps.renyi_entropy(sites, "unrestricted6tensors", chi=1)["cuts"]
        self.assertEqual(len(lengths), 9)
        self.assertEqual(len(entropy), 9)
        self.assertTrue(all(row["xi"] == 0. for row in lengths))
        self.assertTrue(all(abs(x) < 1e-12 for row in entropy for x in row["S2"]))

    def test_nine_entropies_against_original_cylinder(self):
        generator = torch.Generator().manual_seed(20261025)
        sites = tuple(normalize(.2 + torch.rand((2, 2, 2, 2), dtype=torch.float64,
                                               generator=generator)) for _ in range(6))
        result = hive_ipeps.renyi_entropy(sites, "unrestricted6tensors", chi=4, lengths=(4,),
                                   tolerance=1e-11)["cuts"]
        for direction in range(3):
            for cut in range(3):
                self.assertAlmostEqual(result[3*direction+cut]["S2"][0],
                                       cylinder_entropy(sites, cut), delta=2e-8)
            sites = tuple(sites[i].permute(1, 2, 0, 3) for i in (2, 3, 4, 5, 0, 1))

    def test_threefold_reduction(self):
        generator = torch.Generator().manual_seed(15)
        parameters = [normalize(.2+torch.rand((2, 2, 2, 2), dtype=torch.float64,
                                            generator=generator)) for _ in range(2)]
        sites = hive_ipeps.unit_cell(parameters, "c3rotated2tensors")
        for measure, key, value in ((hive_ipeps.renyi_entropy, "cuts", "S2"),
                                    (hive_ipeps.correlation_lengths, "directions", "xi")):
            reduced = measure(parameters, "c3rotated2tensors", chi=4)[key]
            full = measure(sites, "unrestricted6tensors", chi=4)[key]
            for direction in range(3):
                for cut in range(3):
                    np.testing.assert_allclose(full[3*direction+cut][value], reduced[cut][value],
                                               atol=2e-9, rtol=2e-7)

    def test_parallel_mps_chains(self):
        # C-B-E-F repeats along alpha; A,D are nearly isolated product spins.
        mps = torch.tensor([[[.8, 0.], [0., .5]], [[0., .4], [.3, 0.]]],
                           dtype=torch.float64)
        transfer = contract("plr,pLR->lLrR", mps, mps).reshape(4, 4)
        values = torch.linalg.eigvals(transfer).abs().sort(descending=True).values
        expected = float((values[1] / values[0])**4)
        generator = torch.Generator().manual_seed(4)
        sites = []
        for site in range(6):
            tensor = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
            if site in (0, 3):
                tensor[0, 0, 0, 0] = 1.
            else:
                incoming, outgoing = {2: (1, 0), 1: (0, 2), 4: (2, 0), 5: (0, 1)}[site]
                for spin in range(2):
                    for left in range(2):
                        for right in range(2):
                            index = [0, 0, 0, spin]
                            index[incoming], index[outgoing] = left, right
                            tensor[tuple(index)] = mps[spin, left, right]
            # A weak generic coupling keeps every virtual sector populated.
            sites.append(normalize(tensor + 1e-3*torch.rand(tensor.shape,
                                   dtype=tensor.dtype, generator=generator)))
        result = hive_ipeps.correlation_lengths(sites, "unrestricted6tensors", chi=4,
                                         max_steps=150, tolerance=1e-12)["directions"]
        for row in result[:3]:
            self.assertAlmostEqual(row["lambda2_over_lambda1_abs"], expected, delta=.004)
        for row in result[3:]:
            self.assertLess(row["lambda2_over_lambda1_abs"], 1e-6)


if __name__ == "__main__":
    unittest.main()

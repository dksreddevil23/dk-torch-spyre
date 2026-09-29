# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Owner(s): ["module: cpp"]

# NOTE: All tests here should run on their own subprocess
import os

from torch.testing._internal.common_utils import run_tests, TestCase


class TestSpyre(TestCase):
    def test_device_layout_available_before_runtime_init(self):
        """The tensor monkey-patch (`.to(device_layout=)` etc.) must be applied
        at import time, before the device runtime is started.

        This is the contract the autoload/lazy-init split relies on: importing
        ``torch_spyre`` installs ``Tensor.to(device_layout=...)``,
        ``Tensor.device_tensor_layout()`` and ``torch.empty(device_layout=)``
        without calling ``start_runtime()``. Run in a fresh process so the
        result is not contaminated by other tests that already started the
        runtime in this session.
        """
        import sys
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import inspect
            import torch

            # Runtime must NOT have been started just by importing.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized during import"
            )

            # The monkey-patch must already be installed.
            assert getattr(torch.Tensor, "_spyre_tensor_patched", False) is True, (
                "tensor patch not applied at import time"
            )

            # `.to` must accept the device_layout kwarg (i.e. it is spyre_to,
            # not the stock Tensor.to).
            assert "device_layout" in inspect.signature(torch.Tensor.to).parameters, (
                "Tensor.to does not accept device_layout after import"
            )

            # device_tensor_layout() must be installed and callable on a plain
            # CPU tensor without touching the runtime (returns None for non-spyre).
            assert hasattr(torch.Tensor, "device_tensor_layout"), (
                "device_tensor_layout not installed at import time"
            )
            assert torch.ones(2).device_tensor_layout() is None

            # torch.empty must accept device_layout too.
            assert "device_layout" in inspect.signature(torch.empty).parameters, (
                "torch.empty does not accept device_layout after import"
            )

            # Still must not have started the runtime as a side effect.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized while exercising the patched API"
            )

            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"

    def test_manual_seed_does_not_start_runtime(self):
        """torch.manual_seed(seed) must not open a physical device.

        torch.testing._internal.common_utils calls torch.manual_seed(SEED)
        at import time, which cascades into torch.spyre.manual_seed_all().
        Every upstream test file imports common_utils, so if manual_seed_all
        eagerly started the runtime, merely *collecting* (importing) any
        such file would acquire a real Spyre card -- causing spurious
        device-open contention when multiple processes collect concurrently
        (e.g. pytest --collect-only fanned out across cards). The default
        generators are pure host-side RNG state (see SpyreGeneratorImpl) and
        never needed the runtime to be started to be seeded.

        Run in a fresh process so the result is not contaminated by other
        tests that already started the runtime in this session.
        """
        import sys
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import torch
            import torch_spyre  # noqa: F401

            torch.manual_seed(0xC0FFEE)

            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized by torch.manual_seed() alone"
            )

            # The seed must still have taken effect on the default generator.
            assert torch.spyre.initial_seed() == 0xC0FFEE, (
                "seed was not applied to the default generator"
            )

            # Still must not have started the runtime as a side effect of
            # reading it back.
            assert torch.spyre.is_initialized() is False, (
                "runtime was initialized while reading back the seed"
            )

            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        proc = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=130,
            text=True,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"

    def test_lazy_init_without_prior_prepare_does_not_deadlock(self):
        """_lazy_init() must not deadlock when it is the first call to touch
        the runtime in a process.

        _lazy_init() holds _runtime_init_lock and, from within that critical
        section, calls _prepare_c_extension() -- which normally short-circuits
        immediately because manual_seed()/manual_seed_all() (called by
        torch.testing._internal.common_utils at import time) already ran
        _prepare_c_extension() first and populated self._C. But a process
        that never calls any RNG method before its first real device op --
        e.g. a freshly spawned Inductor compile-worker subprocess, which
        does not re-import common_utils -- hits _lazy_init() as the very
        first caller, with self._C still None. If _runtime_init_lock were a
        plain (non-reentrant) Lock, _prepare_c_extension()'s own `with
        _runtime_init_lock:` would then block forever on the lock
        _lazy_init() is still holding. Uses a short timeout so a regression
        fails fast instead of hanging the whole test run.
        """
        import sys
        import subprocess
        import textwrap

        script = textwrap.dedent("""
            import torch
            import torch_spyre  # noqa: F401

            # Call _lazy_init() directly, with nothing having called
            # manual_seed()/_prepare_c_extension() first in this process --
            # reproducing the fresh compile-worker-subprocess ordering.
            torch.spyre._impl._lazy_init()

            assert torch.spyre.is_initialized() is True, (
                "runtime did not report initialized after _lazy_init()"
            )

            print("OK")
        """)

        env = os.environ.copy()
        env["DT_DEEPRT_VERBOSE"] = "-1"
        env["DTLOG_LEVEL"] = "error"

        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=60,
                text=True,
            )
        except subprocess.TimeoutExpired as e:
            raise AssertionError(
                "_lazy_init() deadlocked (timed out) when called without a "
                "prior _prepare_c_extension() call -- check "
                "_runtime_init_lock reentrancy"
            ) from e
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        assert proc.returncode == 0, (
            f"subprocess failed (rc={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}"
        )
        assert out.endswith("OK"), f"unexpected stdout:\n{out}\nstderr:\n{err}"


if __name__ == "__main__":
    run_tests()

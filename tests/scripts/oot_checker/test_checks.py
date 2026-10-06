# Copyright 2026 The Torch-Spyre Authors.
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

"""oot_checker.checks.check_invalid_platforms: typo'd exclude_platforms values."""

from pathlib import Path

from oot_checker.checks import KNOWN_PLATFORMS, check_invalid_platforms


def test_known_value_passes():
    assert check_invalid_platforms({Path("a.yaml"): ["ppc64le"]}) == 0


def test_typo_is_flagged():
    assert check_invalid_platforms({Path("a.yaml"): ["ppc64"]}) == 1
    assert check_invalid_platforms({Path("a.yaml"): ["power"]}) == 1


def test_empty_list_passes():
    assert check_invalid_platforms({Path("a.yaml"): []}) == 0


def test_case_insensitive():
    assert check_invalid_platforms({Path("a.yaml"): ["PPC64LE"]}) == 0


def test_known_platforms_contains_expected_values():
    assert KNOWN_PLATFORMS == {"x86_64", "ppc64le", "s390x", "aarch64"}

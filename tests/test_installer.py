"""Static safety checks for the Windows KataGo installer."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTALLER = PROJECT_ROOT / "安装KataGo.ps1"


class KataGoInstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = INSTALLER.read_text(encoding="utf-8")

    def test_installer_pins_verified_assets_instead_of_latest_release(self) -> None:
        self.assertNotIn("/releases/latest", self.script)
        self.assertIn("katago-v1.18.1-opencl-windows-x64.zip", self.script)
        self.assertIn(
            "1710DB1903AB921AA6837A9599C8474F8A59F057650217C5D9BC125EE393A9FF",
            self.script,
        )
        self.assertIn(
            "0BA27ECED5180B3E3D0B898B280C541112989765E789D1EB6CD0D31B2B2C1229",
            self.script,
        )
        self.assertIn(
            "637746E44F0EFE00AD1245A50AA9BBF0716EFE364C43965EAD97BD6835D84AB5",
            self.script,
        )

    def test_fixed_digest_is_authoritative_over_api_metadata(self) -> None:
        fixed_digest_branch = self.script.index("if ($ExpectedSha256)")
        api_fallback_branch = self.script.index(
            "elseif ($Asset.digest -and $Asset.digest.StartsWith"
        )
        self.assertLess(fixed_digest_branch, api_fallback_branch)
        self.assertIn("$apiDigest -ne $expected", self.script)

    def test_assets_install_into_gitignored_standard_folder(self) -> None:
        self.assertIn('Join-Path $projectRoot "katago"', self.script)
        self.assertNotIn('Join-Path $projectRoot "vendor\\katago"', self.script)


if __name__ == "__main__":
    unittest.main()

# Create GitHub milestones and labels for LFM2.5-JP
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_github_meta.ps1

$ErrorActionPreference = "Stop"
$Repo = "sotanengel/LFM2.5-JP"

$milestones = @(
    @{ title = "Phase 0: Environment"; due = "2026-07-09" },
    @{ title = "Phase 1: Data Pipeline"; due = "2026-07-16" },
    @{ title = "Phase 2: CPT"; due = "2026-08-06" },
    @{ title = "Phase 3: SFT"; due = "2026-08-20" },
    @{ title = "Phase 4: DPO"; due = "2026-08-27" },
    @{ title = "Phase 5: Release"; due = "2026-09-03" }
)

$labels = @(
    @{ name = "epic"; color = "5319E7" },
    @{ name = "phase-0"; color = "0E8A16" },
    @{ name = "phase-1"; color = "1D76DB" },
    @{ name = "phase-2"; color = "FBCA04" },
    @{ name = "phase-3"; color = "D93F0B" },
    @{ name = "phase-4"; color = "B60205" },
    @{ name = "phase-5"; color = "006B75" },
    @{ name = "infrastructure"; color = "C5DEF5" },
    @{ name = "data"; color = "BFDADC" },
    @{ name = "train"; color = "FEF2C0" },
    @{ name = "eval"; color = "E99695" },
    @{ name = "export"; color = "C2E0C6" },
    @{ name = "docs"; color = "0075CA" },
    @{ name = "blocked"; color = "000000" }
)

foreach ($m in $milestones) {
    $dueOn = "$($m.due)T23:59:59Z"
    gh api "repos/$Repo/milestones" -f title=$m.title -f due_on=$dueOn 2>$null
    Write-Host "Milestone: $($m.title)"
}

foreach ($l in $labels) {
    gh label create $l.name --color $l.color --force 2>$null
    Write-Host "Label: $($l.name)"
}

Write-Host "Done."

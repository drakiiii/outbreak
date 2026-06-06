"""Tests for the command-line interface (outbreak.cli).

These call ``cli.main(argv)`` in-process (no subprocess) and check exit codes,
printed output and any files written.
"""

import json

import pytest

from outbreak import cli


def test_list_presets(capsys):
    assert cli.main(["--list-presets"]) == 0
    out = capsys.readouterr().out
    assert "covid_like" in out and "measles_like" in out


def test_single_run_prints_summary(capsys):
    # Small, deterministic and short so the test is fast.
    code = cli.main([
        "--preset", "influenza_like", "--population", "50000",
        "--days", "120", "--no-stochastic",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "influenza_like" in out
    assert "Attack rate" in out or "Total infections" in out


def test_single_run_writes_csv_and_json(tmp_path, capsys):
    csv_path = tmp_path / "ts.csv"
    json_path = tmp_path / "summary.json"
    code = cli.main([
        "--preset", "covid_like", "--population", "40000", "--days", "100",
        "--no-stochastic", "--csv", str(csv_path), "--json", str(json_path),
        "--quiet",
    ])
    assert code == 0
    # CSV: header plus one row per simulated day.
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 101                       # 100 days + header
    assert lines[0].startswith("day,")
    # JSON: a summary dict with the headline metrics.
    summary = json.loads(json_path.read_text())
    assert "attack_rate" in summary and "total_deaths" in summary


def test_quiet_suppresses_output(capsys):
    cli.main(["--preset", "covid_like", "--population", "30000", "--days", "60",
              "--no-stochastic", "--quiet"])
    assert capsys.readouterr().out == ""


def test_agent_engine_runs(capsys):
    code = cli.main([
        "--engine", "agent", "--n-agents", "20000", "--population", "100000",
        "--days", "80", "--initial-infected", "400", "--r0", "2.0", "--seed", "1",
        "--quiet",
    ])
    assert code == 0


def test_network_flag_warns_for_compartmental(capsys):
    # --network is agent-only; with the default compartmental engine it should
    # still run but print a note to stderr.
    code = cli.main(["--population", "30000", "--days", "60", "--no-stochastic",
                     "--network", "--quiet"])
    assert code == 0
    assert "network" in capsys.readouterr().err.lower()


def test_ensemble_reports_and_writes_per_run_csv(tmp_path, capsys):
    csv_path = tmp_path / "ens.csv"
    code = cli.main([
        "--preset", "covid_like", "--population", "30000", "--days", "120",
        "--ensemble", "4", "--seed", "0", "--csv", str(csv_path),
    ])
    assert code == 0
    assert "ensemble" in capsys.readouterr().out.lower()
    # One header row plus one row per run.
    assert len(csv_path.read_text().strip().splitlines()) == 5


def test_scenario_roundtrip_from_file(tmp_path):
    """A scenario saved by the library can be loaded and run via --scenario."""
    from outbreak import preset_scenario
    scenario = preset_scenario("influenza_like", total_population=25000)
    scenario.simulation.duration_days = 60
    scenario.simulation.stochastic = False
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario.to_dict()))

    assert cli.main(["--scenario", str(path), "--quiet"]) == 0


# ------------------------------------------------------------- error handling
def test_missing_scenario_file_returns_error(capsys):
    assert cli.main(["--scenario", "/no/such/file.json"]) == 2
    assert "not found" in capsys.readouterr().err


def test_invalid_population_returns_error(capsys):
    # A negative population fails config validation -> handled, exit code 2.
    assert cli.main(["--population", "-100"]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_invalid_preset_exits():
    # argparse rejects an unknown choice by raising SystemExit.
    with pytest.raises(SystemExit):
        cli.main(["--preset", "does_not_exist"])

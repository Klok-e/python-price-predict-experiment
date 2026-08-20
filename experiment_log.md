# Experiment Log

## Objective and acceptance contract

- Primary objective: maximize Compounded Net Return after all-in turnover cost and actual funding.
- Hard risk constraint: every evidence run must remain within the 20% Drawdown Limit.
- Final proof: at least 60 days and 100 Qualifying Portfolio Changes of fresh Forward Paper Proof,
  with positive net return and no Risk Stop.
- Statistical prediction quality and buy-and-hold outperformance are diagnostics, not proof gates.

## Rejected approaches

| Approach | Durable lesson | Verdict |
| --- | --- | --- |
| One-minute fixed-horizon price prediction | Large target, model, threshold, horizon, and hold grids produced fragile selection surfaces; retained results failed rolling proof after costs. | Rejected |
| Long-only threshold execution | Proxy predictions and forced directional exposure did not optimize the portfolio objective and could not express short or cash conviction. | Rejected |
| Rank and regime selection | Some isolated holdouts looked strong, but results were not stable across chronological regimes and did not establish fresh paper profitability. | Rejected |
| Buy-and-hold excess gates | Beating a benchmark is not the capital objective; a losing strategy can still beat a worse benchmark. | Rejected |
| Stored ledgers and HTML timelines | High-volume artifacts obscured the reproducible evidence contract and consumed unnecessary storage. | Rejected |

## New experiment entry format

Append each actual model experiment below. Infrastructure tests are not experiments.

### YYYY-MM-DD - Short name

- Configuration hash:
- Code hash:
- Data hash:
- Model hash:
- Walk-Forward Compounded Net Return:
- Maximum Drawdown by fold:
- Turnover / costs / funding:
- Historical Holdout or Forward Paper Proof state:
- Verdict:
- Notes:

### 2026-08-03 - Initial ROCm direct-policy protocol

- Configuration hash: `154bbba9f5ca3d3a16921bc4c22c71722d97787fb5f9cb8f1672a87ae4de4f71`
- Code hash: `b726aa5fd1f85e015b95abcdf597df30da4bb4eacc3db24ff055bd5c5a9ba1d0`
- Data hash: `b9b7fe9c2da99f31d0942825462875c0c118e257ed5e028cd80976dede0f7af5`
- Model hash: validation `34d2df59a4051cd24ba23629d6ecb7a547cfd36b8ad656cdde7fe2c390eb578f`;
  holdout prequential fit `0d4b16efbac7452f97ed6f147b0942d71e52bded788976b4f0740ef78a5239d6`
- Walk-Forward Compounded Net Return: +25.541167%; selected actual three-seed TCN ensemble,
  width 64, one-day receptive field; 1,650 Qualifying Portfolio Changes
- Maximum Drawdown by fold: aggregate maximum 6.926840%; the initial report did not emit the
  twelve individual fold values, which is an evidence-contract defect to correct in the revision
- Turnover / costs / funding: not emitted by the initial validation report, which is an
  evidence-contract defect to correct in the revision
- Historical Holdout or Forward Paper Proof state: initial 2026-05-01 through 2026-07-31 holdout
  consumed exactly once; −6.448210% Compounded Net Return, 6.974629% Maximum Drawdown, 73 Qualifying
  Portfolio Changes; no Forward Paper Proof started
- Verdict: validation passed, initial Historical Holdout failed and is permanently Consumed Holdout
- Notes: Holdout result hash
  `f7620544b5580f3a913222f29210069204b34cefe31255661238b589110c79fa`.
  The failed holdout is Development Evidence for subsequent revisions and cannot be reused as final
  proof. Fresh Forward Paper Proof is required for any revised protocol.

### 2026-08-03 - Consumed-holdout development revision

- Configuration hash: `6702f2d904691db169eaed251f9bc59c014ff547ee92e8614aa56e43694f181f`
- Code hash: `4a2bd8b42ccc9e849d2103750808b26550f42849f8fbde89b45d11121062b7aa`
- Data hash: `b9b7fe9c2da99f31d0942825462875c0c118e257ed5e028cd80976dede0f7af5`
- Model hash: `9a0e113926ba4c039400ecd931f59d9660a5c59a7e6efcb7db8242e9522a423a`
- Walk-Forward Compounded Net Return: +19.713728%; selected actual three-seed TCN ensemble,
  width 64, one-day receptive field; 343 Qualifying Portfolio Changes
- Maximum Drawdown by fold: 1.302482%, 3.643021%, 3.521775%, 4.429805%, 2.207709%,
  2.625725%, 1.639066%, 1.157398%, 5.353192%, 3.394632%, 3.571769%, and 3.139223%;
  no Risk Stop triggered
- Turnover / costs / funding: $54,080.80 turnover notional, $37.86 all-in transaction cost, and
  -$121.12 funding cashflow
- Historical Holdout or Forward Paper Proof state: the initial holdout remains permanently consumed;
  this revision includes that known period as Development Evidence and has no Forward Paper Proof yet
- Verdict: Validated Policy Protocol for development; fresh Forward Paper Proof required
- Notes: Validation result hash
  `d5bcc04a90001eabbde7b860dfe311cca7da95c623bf6cd57ba37df0ead0cc53`.
  Fold returns were +2.905684%, +7.753838%, +8.070182%, +0.257837%, +1.481497%, +0.471566%,
  -0.070855%, +2.066533%, +0.849274%, -3.030506%, -0.750526%, and -1.279821%.

### 2026-08-20 - Causal delayed-fill protocol

- Configuration hash: `03722c682b8d35618bc2c5a6ab9c656665a330d6acff2dad38a0c71142136536`
- Code hash: `de5778920ae6ce8e68670de7dd922f6d44152b2865d8d883e914312183e2f51e`
- Data hash: `6557b79293d8498363d1146331e420dd159f912fd01359d5dac7d03ffb478f28`
- Model hash: `599fd50d2713a1a00f722d5c32024036d98831f88d11c435d7c5c2e14c75fc4a`
- Walk-Forward Compounded Net Return: +29.026712%; selected actual three-seed TCN ensemble,
  width 64, seven-day receptive field; 300 Qualifying Portfolio Changes
- Maximum Drawdown by fold: 1.819838%, 7.887010%, 4.268140%, 4.339552%, 2.133313%,
  2.162004%, 1.463445%, 2.371882%, 7.676047%, 5.231659%, 4.796008%, and 3.680513%;
  no Risk Stop triggered
- Turnover / costs / funding: $60,987.06 turnover notional, $42.69 all-in transaction cost, and
  -$221.76 funding cashflow
- Historical Holdout or Forward Paper Proof state: the initial holdout remains permanently
  consumed; this revised protocol has not used a new holdout and requires fresh Forward Paper Proof
- Verdict: Validated Policy Protocol for development; fresh Forward Paper Proof required
- Notes: Validation result hash
  `4a78ca67b4eb60b20ddca90044fbad97bc2e674ef21941438a605de29ae0f721`.
  Fold returns were +4.998817%, +12.600135%, +9.569493%, +0.544924%, +1.509606%, +0.308431%,
  -0.436581%, +3.435139%, +1.710661%, -4.709000%, -1.174082%, and -1.370959%.

### 2026-08-20 - Gap-safe unattended paper protocol

- Configuration hash: `03722c682b8d35618bc2c5a6ab9c656665a330d6acff2dad38a0c71142136536`
- Code hash: `d87bc84f00375040d847cba756dbd175b06a11834f8ea4bf4bb16598176abf73`
- Data hash: `6557b79293d8498363d1146331e420dd159f912fd01359d5dac7d03ffb478f28`
- Model hash: `599fd50d2713a1a00f722d5c32024036d98831f88d11c435d7c5c2e14c75fc4a`
- Walk-Forward Compounded Net Return: +29.026712%; selected actual three-seed TCN ensemble,
  width 64, seven-day receptive field; 300 Qualifying Portfolio Changes
- Maximum Drawdown by fold: 1.819838%, 7.887010%, 4.268140%, 4.339552%, 2.133313%,
  2.162004%, 1.463445%, 2.371882%, 7.676047%, 5.231659%, 4.796008%, and 3.680513%;
  no Risk Stop triggered
- Turnover / costs / funding: $60,987.06 turnover notional, $42.69 all-in transaction cost, and
  -$221.76 funding cashflow
- Historical Holdout or Forward Paper Proof state: the initial holdout remains permanently
  consumed; fresh proof for protocol
  `7bed0db0dc52cd736251e29699b69928c22c2f930a5a9a0f09fcb7e06b91b45c` started flat at
  `2026-08-20T01:22:00+00:00` and advanced through `2026-08-20T01:25:00+00:00` with zero restarts
- Verdict: Validated Policy Protocol; Forward Paper Proof is actively accumulating final evidence
- Notes: immutable validation artifact `73d9227a245d1eba`, result hash
  `4a78ca67b4eb60b20ddca90044fbad97bc2e674ef21941438a605de29ae0f721`.
  The code-only revision bounds contemporaneous mark latency, archives and resets interrupted proof
  attempts from a Flat Start, distinguishes transient publication lag from irreconstructible gaps,
  and runs through a reboot-persistent user service. Historical policy metrics remain bit-identical.

# C-MAPSS RUL LSTM FedAli job

This job adapts the published FedAli Alignment with Prototypes (ALP) algorithm
to the existing LSTM RUL pipeline. The ALP layer processes all LSTM timestep
embeddings, and the aligned final timestep is used for RUL regression.

## Protocol

- Each client starts with a prototype bank distinct from the global bank and
  retains that non-persistent personalized bank across rounds.
- During training, Sinkhorn matching is computed once over the concatenated
  local and global banks and partitioned into local and global assignments.
- Each local prototype selects `ceil((batch * sequence_length) / G)` embeddings
  and is updated from their assignment-weighted mean using EMA.
- Local prototypes are not optimized by backpropagation.
- Aligned embeddings are L2-normalized after GLU regulation and interpolation.
- Clients upload full model state with their local bank in
  `alp.global_prototypes`.
- The server applies sample-weighted FedAvg to ordinary model tensors and
  weighted K-means to all uploaded prototype vectors.
- The clustered centers become the global bank for the next round.
- Local evaluation uses personalized local prototypes. A standalone global
  checkpoint has no local memory and therefore evaluates with global prototypes.

## Defaults

| Setting | Environment variable | Default |
|---|---|---:|
| Number of prototypes | `FEDALI_NUM_PROTOTYPES` | 16 |
| Alignment coefficient | `FEDALI_BETA` | 0.2 |
| Local EMA coefficient | `FEDALI_GAMMA` | 0.999 |
| Sinkhorn temperature | `FEDALI_EPSILON` | 0.05 |
| Sinkhorn iterations | `FEDALI_SINKHORN_ITERATIONS` | 3 |

The defaults for beta, gamma, epsilon, and Sinkhorn iterations follow the
published FedAli experiment. The single-bank prototype count of 16 is specific
to this LSTM RUL adaptation and should be tuned using validation cycles only.

The job uses full-model transfer (`WEIGHTS`) rather than the `WEIGHT_DIFF`
transfer used by the FedAvg and FedProx jobs. This is intentional and required
for prototype-bank clustering.

The original paper places an ALP layer in every transformer encoder block. This
job retains the project's LSTM backbone, so it uses one ALP layer after the
LSTM. It follows the published ALP and server aggregation mechanics but is an
architecture adaptation rather than a reproduction of the paper's HART model.

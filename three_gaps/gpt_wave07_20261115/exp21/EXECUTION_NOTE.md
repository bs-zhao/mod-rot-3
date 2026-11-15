# Query throughput revision

The scalar-query implementation completed hybrid h=.05/.1/.25 in CR3 rep1,
about255 seconds each, including calibration, search and diagnostic landscapes.
These completed results remain intact under independent_v1. The next scalar
configuration was interrupted for a query-throughput revision; incomplete
folders have no COMPLETE marker and are not scientific results.

The vectorized implementation uses exactly the same physical four-corner
queries for EVERY candidate R. It evaluates them in row batches and reuses the
identical (i,k)/(k,i) UU stencil, preserving both matrix entries in UUd_share.
It never treats finite differences as a pre-rotatable tensor. Mathematical and
actual-decoder energy/gradient agreement are checked before a separate
fd_vectorized namespace is executed. This revision is not a change of h, loss,
measure, covariance, starts, steps or parameter-based selection.

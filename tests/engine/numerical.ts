export function assertClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string, atol = 2e-5, rtol = 1e-5): number {
  if (actual.length !== expected.length) throw new Error(`${label}: mismatched lengths ${actual.length}/${expected.length}`);
  let maximum = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || error > atol + rtol * Math.abs(expected[i]))
      throw new Error(`${label}[${i}]: actual=${actual[i]}, expected=${expected[i]}, error=${error}`);
    maximum = Math.max(maximum, error);
  }
  return maximum;
}

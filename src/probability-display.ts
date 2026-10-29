/** Keep small, nonzero effects visible instead of rounding them to zero. */
export function compactPercent(probability: number): string {
  const value = probability * 100;
  return Math.abs(value) >= .1 || value === 0 ? value.toFixed(1) : value.toPrecision(2);
}

export function probabilityDirection(delta: number): string {
  return delta > 0 ? 'More likely' : delta < 0 ? 'Less likely' : 'Probability unchanged';
}

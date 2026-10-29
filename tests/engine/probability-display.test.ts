import { expect, test } from 'vitest';
import { compactPercent, probabilityDirection } from '../../src/probability-display';

test('compact percentages preserve small nonzero values and distinguish them from measured zero', () => {
  expect(compactPercent(.704064)).toBe('70.4');
  expect(compactPercent(.154204)).toBe('15.4');
  expect(compactPercent(.00000046)).toBe('0.000046');
  expect(compactPercent(1e-12)).toBe('1.0e-10');
  expect(compactPercent(0)).toBe('0.0');
});

test('effect direction uses the unrounded signed measurement, including tiny effects and identity', () => {
  expect(probabilityDirection(1e-12)).toBe('More likely');
  expect(probabilityDirection(-1e-12)).toBe('Less likely');
  expect(probabilityDirection(0)).toBe('Probability unchanged');
});

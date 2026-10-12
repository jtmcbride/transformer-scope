import { readFileSync, writeFileSync } from 'node:fs';
const report = JSON.parse(readFileSync('.cache/browser-benchmark-results.json', 'utf8'));
const records = [];
function visit(suites) {
  for (const suite of suites) {
    for (const spec of suite.specs ?? []) for (const test of spec.tests) for (const result of test.results)
      if (result.status === 'passed') for (const attachment of result.attachments) if (attachment.name === 'browser-benchmark')
        records.push(JSON.parse(attachment.body ? Buffer.from(attachment.body, 'base64').toString('utf8') : readFileSync(attachment.path, 'utf8')));
    visit(suite.suites ?? []);
  }
}
visit(report.suites);
writeFileSync('.cache/browser-benchmark.json', JSON.stringify({ recordedAt: new Date().toISOString(), records }, null, 2) + '\n');
console.log(`Collected ${records.length} successful browser benchmark records.`);

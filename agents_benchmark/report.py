"""Report."""
import csv
from io import StringIO
from pathlib import Path
from .catalog import TASKS
from .config import MODEL
from .scoring import score_group
from .storage import read_json, atomic_json

from .protocol import ARMS, cohort_digest, run_identity

def paired_answers(samples, output):
    answers = {arm: [] for arm in ARMS}
    cohort = cohort_digest(samples)
    for sample in samples:
        for arm in ARMS:
            directory = sample.directory(Path(output) / arm)
            expected = run_identity(sample, arm, cohort)
            if read_json(directory / 'input_identity.json') != expected:
                raise ValueError(f'Missing or mismatched input identity: {arm}/{sample.task}/{sample.sample_id}')
            outcome = read_json(directory / 'outcome.json')
            if outcome is None:
                raise ValueError(f'Incomplete three-arm comparison: {arm}/{sample.task}/{sample.sample_id}')
            if outcome.get('status') == 'protocol_error' or outcome.get('model', MODEL) != MODEL:
                raise ValueError('A protocol mismatch cannot be included in the comparison')
            # After all five attempts (Supplementary Material S.1-E) an unresolved
            # failure stays in the evaluation and is scored by the failure rules.
            if outcome.get('status') not in {'completed', 'incomplete'} and not outcome.get('retry_exhausted'):
                raise ValueError(f"Cannot score infrastructure or unresolved failure: {arm}/{sample.task}/{sample.sample_id}: {outcome.get('status')}")
            if outcome.get('fingerprint', sample.fingerprint) != sample.fingerprint:
                raise ValueError('Outcome belongs to a different input')

            answers[arm].append(outcome.get('answer_text', '') if outcome['status'] == 'completed' else None)
    return answers


def write_report(samples, output):
    answers = paired_answers(samples, output)
    rows = []
    for metric, task in TASKS.items():
        indices = [i for i, sample in enumerate(samples) if sample.task == task]
        if not indices:
            raise ValueError(f'Missing benchmark task: {task}')
        cohort = [samples[i] for i in indices]
        scores = {arm: score_group(cohort, [answers[arm][i] for i in indices], metric) for arm in ARMS}
        rows.append({'task':metric, **{arm:scores[arm]['score'] for arm in ARMS},
                     'n':len(cohort),
                     **{arm + '_valid_rate':scores[arm]['valid_rate'] for arm in ARMS},
                     **{arm + '_mae':scores[arm].get('mae') for arm in ARMS},
                     'autonomous_minus_direct':scores['autonomous']['score']-scores['direct']['score'],
                     'guided_minus_autonomous':scores['guided']['score']-scores['autonomous']['score']})
    atomic_json(Path(output) / 'comparison.json', rows)
    table = StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (Path(output) / 'comparison.csv').write_text(table.getvalue())
    return rows

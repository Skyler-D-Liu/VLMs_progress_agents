"""Protocol."""
import hashlib
import json
from .config import MODEL, build_hosted_request, zero_shot_system_message
from .guided import GUIDANCE

ARMS = ('direct', 'autonomous', 'guided')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cohort_digest(samples):
    identities = sorted((sample.task, sample.sample_id, sample.fingerprint) for sample in samples)
    if not identities or len({(task, sid) for task, sid, _ in identities}) != len(identities):
        raise ValueError('Empty or duplicate cohort identities')
    return digest(identities)


def run_identity(sample, arm, cohort):
    if arm not in ARMS:
        raise ValueError('Unknown comparison arm')
    if arm == 'guided' and sample.task not in GUIDANCE:
        raise ValueError('Guidance is required for every guided task')
    settings = {'model':MODEL, 'protocol':'dataset-paired-v4-stratified-shared-system', 'arm':arm}
    if arm == 'direct':
        settings['system_message'] = zero_shot_system_message(sample.row['system_message'])
        settings['api'] = 'chat.completions'
    else:
        request = build_hosted_request('', sample.row['system_message'])
        settings.update(agent=request['agent'], environment=request['environment'],
                        guidance=GUIDANCE[sample.task] if arm == 'guided' else None)
    return {'fingerprint':sample.fingerprint, 'arm':arm, 'model':MODEL,
            'cohort_sha256':cohort, 'settings_sha256':digest(settings)}

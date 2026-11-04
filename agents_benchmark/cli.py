"""Cli."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .catalog import load_frozen_catalog, prepare_catalog
from .config import DATA_ROOT, MODEL, initialize_key_file, make_client, read_key, zero_shot_system_message
from .guided import GUIDANCE
from .images import ImageStore
from .report import write_report
from .protocol import cohort_digest, run_identity
from .session import SessionRunner
from .storage import api_error, atomic_json, exclusive_lock, read_json, redact


def direct_answer(sample, images, directory, prompt=None):
    from vleo_runtime.api import openai_request
    from evaluation.utils.response_protocol import transient_error
    saved = read_json(directory / 'outcome.json')
    if saved is not None:
        return saved
    intent = directory / 'request.json'
    if intent.exists():
        raise RuntimeError('Direct request already submitted without a saved outcome; inspect the original run')
    atomic_json(intent, {'fingerprint': sample.fingerprint})
    parts = []
    for image in images:
        mime = {'.png':'image/png', '.jpg':'image/jpeg', '.webp':'image/webp'}[image.suffix]
        uri = 'data:' + mime + ';base64,' + base64.b64encode(image.data).decode()
        parts.append({'type':'image_url', 'image_url':{'url':uri}})
    parts.append({'type':'text', 'text':sample.prompt if prompt is None else prompt})
    payload = {'model':MODEL, 'messages':[
        {'role':'system','content':zero_shot_system_message(sample.row['system_message'])},
        {'role':'user','content':parts}]}
    try:
        response = openai_request(payload, read_key())
    except Exception as exc:
        if not transient_error(exc):
            raise
        # Transient failure without a model response (network error, timeout, 429/5xx).
        outcome = {'status':'request_failed', 'answer_text':'', 'error':api_error(exc),
                   'fingerprint':sample.fingerprint}
        atomic_json(directory / 'outcome.json', outcome)
        return outcome
    if response.get('model') != MODEL:
        atomic_json(directory / 'outcome.json', {'status':'protocol_error', 'answer_text':'',
                    'model':response.get('model'), 'fingerprint':sample.fingerprint})
        raise ValueError('Direct API returned a different model')
    choice = response['choices'][0]
    outcome = {'status':'completed' if choice.get('finish_reason') == 'stop' else 'incomplete',
               'answer_text':choice['message'].get('content') or '', 'usage':response.get('usage'),
               'fingerprint':sample.fingerprint, 'model':response['model'],
               'response_id':response.get('id'), 'created_at':response.get('created'),
               'system_fingerprint':response.get('system_fingerprint'),
               'finish_reason':choice.get('finish_reason')}
    atomic_json(directory / 'outcome.json', outcome)
    return outcome


def attempt_prompt(sample, repair):
    """Task prompt, followed by the format-only repair instruction when the
    previous response's final answer failed format validation."""
    from evaluation.utils.response_protocol import repair_instruction
    if not repair:
        return sample.prompt
    return sample.prompt + '\n\n' + repair_instruction(sample.task, sample.row)


def classify_attempt(sample, outcome):
    """'valid' | 'invalid' (a response whose final answer fails validation) |
    'no_response' (transient failure that returned no model answer)."""
    from evaluation.utils.response_protocol import sample_format_valid
    status = outcome.get('status')
    if status == 'completed':
        return 'valid' if sample_format_valid(sample, outcome.get('answer_text')) else 'invalid'
    if status == 'incomplete':
        return 'invalid'
    return 'no_response'


def run_with_retries(sample, directory, attempt):
    """Shared retry rule (Supplementary Material S.1-E): up to five attempts per
    sample. A failure without a model response is retried with the same prompt;
    an invalid final answer is followed by a format-only repair prompt; the first
    valid answer is accepted. Each attempt is stored under attempts/<k>/."""
    from evaluation.utils.response_protocol import MAX_ATTEMPTS
    repair, history, outcome, kind = False, [], {}, 'no_response'
    for number in range(1, MAX_ATTEMPTS + 1):
        prompt = attempt_prompt(sample, repair)
        outcome = attempt(directory / 'attempts' / str(number), prompt)
        if outcome.get('status') in {'api_rejected', 'protocol_error'}:
            raise RuntimeError(f"Attempt {number} ended with {outcome['status']}; inspect it before rerunning")
        kind = classify_attempt(sample, outcome)
        history.append({'attempt':number, 'status':outcome.get('status'), 'repair_prompt':repair,
                        'format_valid':kind == 'valid', 'usage':outcome.get('usage'),
                        'elapsed_seconds':outcome.get('elapsed_seconds')})
        if kind == 'valid':
            break
        # Invalid answer -> repair prompt; no model response -> same task prompt.
        repair = kind == 'invalid'
    final = dict(outcome)
    final.update(attempt_count=len(history), attempts=history, retry_exhausted=kind != 'valid')
    atomic_json(directory / 'outcome.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init','prepare','run','score'))
    parser.add_argument('--output', type=Path, default=Path('results/agents/comparison'))
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-per-task', type=int, help='Maximum samples per task during prepare')
    parser.add_argument('--seed', type=int, default=0, help='Sampling seed during prepare')
    parser.add_argument('--arm', choices=('direct','autonomous','guided'), default='autonomous')
    args = parser.parse_args(argv)
    if args.command == 'init':
        initialize_key_file()
        return
    if args.workers < 1:
        parser.error('--workers must be positive')
    with exclusive_lock(args.output / 'client.lock'):
        if args.command == 'prepare':
            if args.max_per_task is None:
                parser.error('prepare requires --max-per-task')
            prepare_catalog(args.output, args.max_per_task, args.data_root, args.seed)
            return
        samples = load_frozen_catalog(args.output)
        if args.command == 'score':
            write_report(samples, args.output)
            return
        store = ImageStore(args.output)
        cohort = cohort_digest(samples)
        def run(sample, client=None):
            directory = sample.directory(args.output / args.arm)
            outcome = read_json(directory / 'outcome.json')
            settings = read_json(directory / 'input_identity.json')
            identity = run_identity(sample, args.arm, cohort)
            if settings is not None and settings != identity:
                raise ValueError('Saved run input or arm differs from the immutable cohort')
            if outcome is not None:
                if settings != identity:
                    raise ValueError('Saved result lacks matching input identity')
                return outcome
            if (directory / 'state.json').exists() or (directory / 'request.json').exists():
                raise RuntimeError('Sample has an unfinished single-attempt run from an older layout; inspect it before rerunning')
            atomic_json(directory / 'input_identity.json', identity)
            images = store.load(sample)
            if args.arm == 'direct':
                return run_with_retries(sample, directory,
                                        lambda folder, prompt: direct_answer(sample, images, folder, prompt))
            guidance = GUIDANCE.get(sample.task) if args.arm == 'guided' else None
            return run_with_retries(sample, directory, lambda folder, prompt: SessionRunner(
                client, folder, sample.row['system_message'], guidance=guidance).run(prompt, images))
        if args.arm == 'direct':
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(run, samples))
        else:
            with make_client() as client, ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda sample: run(sample, client), samples))


def entrypoint():
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'{type(exc).__name__}: {redact(str(exc))}', flush=True)
        raise SystemExit(1) from None

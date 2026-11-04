"""Scoring."""
from collections import Counter
import importlib.util
import json
import math
import re
from statistics import mean
from agents_benchmark.catalog import Sample, options_from_prompt
from agents_benchmark.config import ROOT
COUNT_KEYS = ('count_before', 'no_damage', 'minor_damage', 'major_damage', 'destroyed')
SPECIES = ('elephants', 'zebras', 'giraffes')
FPCD_OPTIONS = {'A': 'Farm Pond Demolished', 'B': 'Farm Pond Wetted', 'C': 'Farm Pond Dried', 'D': 'No Change', 'E': 'Farm Pond Constructed'}
FPCD_ANSWER_KEYS = {'answer', 'change', 'change_type', 'option', 'description'}
REFUSAL_RE = re.compile(r"\b(?:cannot|can't|unable to|won't|refuse to|not able to)\b", re.I)

def json_answer(text: str):
    from evaluation.utils.cot_prompting import strip_reasoning
    text = strip_reasoning(text)
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)[:-3].strip()
    try:
        return json.loads(text)
    except ValueError:
        decoder = json.JSONDecoder()
        answers, position = [], 0
        while position < len(text):
            if text[position] not in '{[':
                position += 1
                continue
            try:
                value, end = decoder.raw_decode(text, position)
                answers.append(value)
                position = end
            except ValueError:
                position += 1
        if not answers:
            raise ValueError('No JSON answer')
        return answers[-1]


def scalar_count(text, key='count'):
    try:
        value = json_answer(text)
    except ValueError:
        if REFUSAL_RE.search(str(text)):
            return None
        from evaluation.utils.cot_prompting import strip_reasoning
        lines = [line.strip().strip('*` ') for line in strip_reasoning(text).splitlines() if line.strip()]
        candidates = [line for line in lines if re.fullmatch(r'(?:answer|count|total)\s*[:=]\s*.+|[-+]?\d+(?:\.\d+)?', line, re.I)]
        value = candidates[-1] if candidates else (lines[-1] if lines else '')
        match = re.fullmatch(r'(?:answer|count|total)\s*[:=]\s*(.+)', value, re.I)
        if match:
            value = match[1].strip()
    if isinstance(value, dict):
        aliases = {key.casefold(), 'count'}
        if key == 'Answer':
            aliases.update({'answer', 'aircraft', 'aircraft_count', 'aircraft count'})
        values = [v for k, v in value.items() if str(k).strip().casefold() in aliases]
        value = values[0] if len(values) == 1 else None
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', value):
            value = float(value)
        else:
            units = dict(zip(('zero','one','two','three','four','five','six','seven','eight','nine','ten',
                             'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'), range(20)))
            tens = dict(zip(('twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'), range(20,100,10)))
            words = value.casefold().replace('-', ' ').split()
            if len(words) == 1:
                value = units.get(words[0], tens.get(words[0]))
            elif len(words) == 2 and words[0] in tens and words[1] in units and 0 < units[words[1]] < 10:
                value = tens[words[0]] + units[words[1]]
            else:
                value = None
    return nonnegative_integer(value)


def nonnegative_integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)

def category(value: str, options: list[str]):
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold().replace('_', ' ')
    for option in options:
        if normalized == option.casefold():
            return option
    return None

def fpcd_option(value):
    if isinstance(value, str):
        answer = value.strip()
        if answer.upper() in FPCD_OPTIONS:
            return answer.upper()
        for letter, name in FPCD_OPTIONS.items():
            if answer.casefold() == name.casefold():
                return letter
            if re.fullmatch(f'{letter}\\s*[.)\\-:]\\s*{re.escape(name)}', answer, re.I):
                return letter
        return None
    if not isinstance(value, dict) or not value:
        return None
    candidates = [item for key, item in value.items() if key.strip().casefold().replace('-', '_').replace(' ', '_') in FPCD_ANSWER_KEYS]
    if not candidates:
        if len(value) != 1:
            return None
        candidates = list(value.values())
    parsed = {fpcd_option(item) if isinstance(item, str) else None for item in candidates}
    return next(iter(parsed)) if len(parsed) == 1 and None not in parsed else None

def count_fields(json_string):
    from evaluation.utils.cot_prompting import strip_reasoning
    response_text = strip_reasoning(json_string)

    def _coerce_int(value) -> int:
        if value is None or isinstance(value, bool):
            raise ValueError('count must be a non-negative integer')
        if isinstance(value, str):
            value = value.strip()
        number = float(value)
        if not math.isfinite(number) or number < 0 or (not number.is_integer()):
            raise ValueError('count must be a finite non-negative integer')
        return int(number)

    def _canonical_count_key(key):
        key_text = str(key).strip().lower()
        normalized = re.sub('[\\s\\-]+', '_', key_text)
        if normalized in COUNT_KEYS:
            return normalized
        if 'before' in key_text:
            return 'count_before'
        if 'destroyed' in key_text:
            return 'destroyed'
        if 'major' in key_text:
            return 'major_damage'
        if 'minor' in key_text:
            return 'minor_damage'
        if re.search('\\bno\\b|no[_\\-\\s]?damage', key_text):
            return 'no_damage'
        return None
    candidates = [match.group(0) for match in re.finditer('\\{[^{}]*\\}', response_text, re.DOTALL)]
    for candidate in reversed(candidates):
        try:
            parsed_dict = json.loads(candidate)
            if not isinstance(parsed_dict, dict):
                continue
            return_dict = {}
            for key, value in parsed_dict.items():
                canonical_key = _canonical_count_key(key)
                if canonical_key:
                    try:
                        return_dict[canonical_key] = _coerce_int(value)
                    except (TypeError, ValueError, OverflowError):
                        return_dict[canonical_key] = None
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError):
            continue
        if return_dict:
            return return_dict
    return -1

def category_answer(text, aliases):
    options = sorted(set(aliases.values()))
    mapping = {re.sub(r'[^a-z0-9]', '', k.casefold()): v for k, v in aliases.items()}
    mapping.update({re.sub(r'[^a-z0-9]', '', v.casefold()): v for v in options})
    def normalize(value):
        if isinstance(value, bool) or value is None:
            return None
        value = str(value).strip().strip('*` ')
        if value.isdigit():
            index = int(value) - 1
            return options[index] if 0 <= index < len(options) else None
        return mapping.get(re.sub(r'[^a-z0-9]', '', value.casefold()))
    try:
        value = json_answer(text)
    except ValueError:
        lines = [line.strip().strip('*` ') for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            line = re.sub(r'^(?:final\s+answer|answer|category)\s*[:=]\s*', '', line, flags=re.I)
            normalized = normalize(line)
            if normalized is not None:
                return normalized
        matches = [(m.start(), m.end(), v) for k, v in {**aliases, **{v: v for v in options}}.items()
                   for m in re.finditer(r'(?<![a-z0-9])' + re.escape(k.replace('_', ' ')) + r'(?![a-z0-9])', text, re.I)]
        return max(matches)[2] if matches else None
    return normalize(value.get('category')) if isinstance(value, dict) else normalize(value)


def landmark_answer(text, options):
    if not options:
        return None
    try:
        value = json_answer(text)
    except ValueError:
        if REFUSAL_RE.search(str(text)):
            return None
        from evaluation.utils.cot_prompting import strip_reasoning
        lines = [line.strip().strip('*` ') for line in strip_reasoning(text).splitlines()
                 if line.strip() and not line.strip().startswith('```')]
        for line in reversed(lines):
            value = _landmark_value(line, options)
            if value is not None:
                return value
        return None
    if isinstance(value, dict):
        values = [v for k, v in value.items() if str(k).casefold() in {'answer', 'option', 'name', 'landmark'}]
        value = values[0] if len(values) == 1 else None
    return _landmark_value(value, options)


def _landmark_value(value, options):
    if isinstance(value, bool) or value is None:
        return None
    value = str(value).strip().strip('*` ')
    match = re.fullmatch(r'(?:answer\s*[:=]\s*|option\s*)?(\d+)[.)]?(?:\s+(.+))?', value, re.I)
    if match:
        index = int(match[1]) - 1
        if 0 <= index < len(options) and (not match[2] or category(match[2], options) == options[index]):
            return options[index]
        return None
    return category(re.sub(r'^answer\s*[:=]\s*', '', value, flags=re.I), options)


def parse(sample: Sample, text: str):
    from evaluation.utils.cot_prompting import strip_reasoning
    text = strip_reasoning(text)
    task = sample.task
    try:
        try:
            json_answer(text)
        except ValueError:
            if REFUSAL_RE.search(text):
                return None
        if task == 'landmarks':
            return landmark_answer(text, sample.row['options'])
        if task in {'dior_classification', 'patternnet'}:
            from evaluation.utils.task_categories import DIOR_CATEGORIES, PATTERN_CATEGORIES
            aliases = DIOR_CATEGORIES if task == 'dior_classification' else PATTERN_CATEGORIES
            return category_answer(text, aliases)
        if task in {'fair1m', 'dota'}:
            from evaluation.utils.geobench_spatial import parse_fair1m_response, parse_bbox_response
            parsed = (parse_fair1m_response if task == 'fair1m' else parse_bbox_response)(text, sample.row['image_width'], sample.row['image_height'])
            return parsed['bboxes'] if parsed['format_valid'] else None
        if task == 'fpcd':
            try:
                return fpcd_option(json_answer(text))
            except ValueError:
                lines = [line.strip().strip('*` ') for line in text.splitlines() if line.strip()]
                return next((answer for line in reversed(lines) if (answer := fpcd_option(line)) is not None), None)
        if task == 'rareplanes':
            try:
                return scalar_count(text, 'Answer')
            except ValueError:
                return None
        if task in {'cowc', 'neon'}:
            return scalar_count(text)
        if task == 'xview2':
            counts = count_fields(text)
            return {key: counts.get(key) for key in COUNT_KEYS} if isinstance(counts, dict) else None
        value = json_answer(text)
        if task in {'patternnet', 'dior_classification'}:
            return category(value.get('category'), options_from_prompt(sample.prompt)) if isinstance(value, dict) else None
        if task in {'xview2', 'animals'}:
            keys = COUNT_KEYS if task == 'xview2' else SPECIES
            return {key: nonnegative_integer(value.get(key)) for key in keys} if isinstance(value, dict) else None
        if task == 'dior_localization':
            if isinstance(value, dict):
                if ('bbox' in value) == ('bboxes' in value):
                    return None
                value = value.get('bbox', value.get('bboxes'))
            if not isinstance(value, list):
                return None
            if len(value) == 4 and all((isinstance(v, (int, float)) for v in value)):
                value = [value]
            boxes = []
            for box in value:
                if not isinstance(box, list) or len(box) != 4 or (not all((type(v) in {int, float} and math.isfinite(v) for v in box))):
                    return None
                if box[0] < 0 or box[1] < 0 or box[0] >= box[2] or (box[1] >= box[3]):
                    return None
                if ((sample.row.get('image_width') is not None and box[2] > sample.row['image_width']) or
                    (sample.row.get('image_height') is not None and box[3] > sample.row['image_height'])):
                    return None
                boxes.append([float(v) for v in box])
            return boxes
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
        return None
    return None

def truth(sample: Sample):
    row, task = (sample.row, sample.task)
    if task == 'landmarks':
        return row['name']
    if task in {'patternnet', 'dior_classification'}:
        label = row['label'].replace('_', ' ').strip()
        if task == 'patternnet':
            aliases = {'dense residential': 'Residential', 'sparse residential': 'Residential', 'ferry terminal': 'Harbor', 'parking lot': 'Parking Space', 'runway marking': 'Runway', 'golf course': 'Golf Course'}
            label = aliases.get(label.casefold(), label)
        result = category(label, options_from_prompt(sample.prompt))
        if result is None:
            raise ValueError(f'Unrecognized ground-truth category: {label}')
        return result
    if task in {'rareplanes', 'cowc'}:
        return row['label']
    if task == 'neon':
        return row['count']
    if task == 'animals':
        counts = Counter((name.lower() for name in row['objects']['categories']))
        return {plural: counts[plural[:-1]] for plural in SPECIES}
    if task == 'xview2':
        return {key: row['gt'][key] for key in COUNT_KEYS}
    if task == 'fpcd':
        return row['ground_truth']
    if task == 'dior_localization':
        return [row['object']['bbox']]
    if task in {'dota', 'fair1m'}:
        if task == 'fair1m':
            from vleo_runtime.localization import polygon_boxes
            return polygon_boxes(sample.row['ground_truth_polygons'])
        return row['target_boxes']
    raise ValueError(task)

def squared_pearson(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError('Unpaired counting values')
    if len(actual) < 2:
        return 0.0
    ax, ay = (mean(actual), mean(predicted))
    dx, dy = ([x - ax for x in actual], [y - ay for y in predicted])
    denominator = sum((x * x for x in dx)) * sum((y * y for y in dy))
    return min(1.0, sum((x * y for x, y in zip(dx, dy))) ** 2 / denominator) if math.isfinite(denominator) and denominator > 0 else 0.0
_spatial = None

def matching_iou(predicted, expected, oriented=False):
    global _spatial
    if _spatial is None:
        path = ROOT / 'evaluation/utils/geobench_spatial.py'
        spec = importlib.util.spec_from_file_location('agents_spatial_metrics', path)
        _spatial = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_spatial)
    matches = _spatial.hungarian_match(predicted, expected, _spatial.polygon_iou if oriented else None)
    return sum((score for _, _, score in matches)) / max(len(predicted), len(expected), 1)

def score_group(samples: list[Sample], answers: list[str | None], metric: str) -> dict:
    if len(samples) != len(answers):
        raise ValueError('Unpaired samples and answers')
    if not samples or len({sample.task for sample in samples}) != 1:
        raise ValueError('Scoring requires a nonempty single-task cohort')
    parsed = [parse(sample, answer) if answer is not None else None for sample, answer in zip(samples, answers)]
    targets = [truth(sample) for sample in samples]
    n = len(samples)
    task = samples[0].task
    if task in {'landmarks', 'patternnet', 'dior_classification', 'fpcd'}:
        valid = [p is not None for p in parsed]
        score = mean((p == t for p, t in zip(parsed, targets)))
        result = {'score': score, 'metric': 'accuracy'}
    elif task in {'dior_localization', 'dota', 'fair1m'}:
        valid = [p is not None for p in parsed]
        oriented = task == 'fair1m'
        scores = [matching_iou(p or [], t, oriented=oriented) for p, t in zip(parsed, targets)]
        result = {'score': mean(scores), 'metric': 'matched_mIoU'}
    else:
        keys = (COUNT_KEYS[1:] if 'Change' in metric else COUNT_KEYS[:1]) if task == 'xview2' else None
        if keys:
            valid = [p is not None and all((p.get(k) is not None for k in keys)) for p in parsed]
            correlations, maes = ([], [])
            for key in keys:
                actual = [t[key] for t in targets]
                prediction = [p[key] if p and p.get(key) is not None else 0 for p in parsed]
                correlations.append(squared_pearson(actual, prediction))
                maes.append(mean((abs(a - b) for a, b in zip(actual, prediction))))
            score = mean(correlations) if all((v is not None for v in correlations)) else None
            result = {'score': score, 'metric': 'Pearson_r_squared', 'mae': mean(maes), 'per_field_r_squared': dict(zip(keys, correlations))}
        else:
            if task == 'animals':
                valid = [p is not None and all((v is not None for v in p.values())) for p in parsed]
                actual = [sum(t.values()) for t in targets]
                prediction = [sum(p.values()) if ok else 0 for p, ok in zip(parsed, valid)]
            else:
                valid = [p is not None for p in parsed]
                actual = targets
                prediction = [p if p is not None else 0 for p in parsed]
            result = {'score': squared_pearson(actual, prediction), 'metric': 'Pearson_r_squared', 'mae': mean((abs(a - b) for a, b in zip(actual, prediction)))}
    result.update(valid_rate=sum(valid) / n)
    return result

def direct_usage(row: dict):
    response = row.get('response')
    usage = response.get('usage') if isinstance(response, dict) else None
    if not usage:
        return None
    return {'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'), 'input_tokens_details': {'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens')}}

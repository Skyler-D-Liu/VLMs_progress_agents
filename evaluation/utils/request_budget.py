"""Request budget."""
import json
import os
from pathlib import Path
from threading import Lock

from reasoning_effort import RequestFailed


GPT_REQUEST_LIMIT = 5


class RetryLimitReached(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, path, records, sample_ids, limit=GPT_REQUEST_LIMIT):
        self.path = Path(path)
        self.limit = min(5, limit)
        self.lock = Lock()
        self.sample_ids = set(sample_ids)
        self.counts = json.loads(self.path.read_text()) if self.path.exists() else {}
        if not isinstance(self.counts, dict) or set(self.counts) - set(sample_ids):
            raise ValueError('Request budget contains unexpected sample identities')
        if any(type(n) is not int or n < 0 for n in self.counts.values()):
            raise ValueError('Invalid persisted request count')
        for sid, row in records.items():

            known = row.get('request_attempts_used', 0)
            if row.get('status') == 'request_failed':
                known = max(known, row.get('attempt_count', 0))
            self.counts[sid] = max(self.counts.get(sid, 0), known)

    def used(self, sid):
        with self.lock:
            return self.counts.get(sid, 0)

    def exhausted(self, sid):
        return self.used(sid) >= self.limit

    def reserve(self, sid):
        with self.lock:
            if sid not in self.sample_ids:
                raise ValueError('Request outside fixed sample cohort')
            count = self.counts.get(sid, 0)
            if count >= self.limit:
                raise RequestFailed(f'Per-sample request limit ({self.limit}) reached')
            counts = {**self.counts, sid: count + 1}
            temporary = self.path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                json.dump(counts, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            self.counts = counts

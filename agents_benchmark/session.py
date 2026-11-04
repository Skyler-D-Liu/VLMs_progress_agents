"""Session."""
from pathlib import Path
import time

from .config import MODEL, agent_instructions, build_hosted_request, build_input
from .storage import api_error, as_dict, atomic_json, clean_evidence, read_json
from .transport import encoded_part, input_manifest, setup_command

TERMINAL = {"completed", "failed", "cancelled"}
FATAL_SERVICE_CODES = {
    "insufficient_quota", "model_not_found", "permission_denied", "invalid_api_key",
    "authentication_error", "resource_not_found", "credit_balance_exhausted",
    "project_spend_limit_exceeded", "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded", "usage_limit_exceeded",
}


class SessionInterrupted(RuntimeError):
    pass


class ProtocolError(RuntimeError):
    pass


class EnvironmentSetupFailed(RuntimeError):

    def __init__(self, environment):
        super().__init__("Hosted input environment failed before task submission")
        self.environment = environment


def validate_session(session: dict, system_message: str) -> None:
    agent, environment = session.get("agent") or {}, session.get("environment") or {}
    if (agent.get("model") != MODEL or agent.get("instructions") != agent_instructions(system_message) or
            (agent.get("multi_agent") or {}).get("enabled") is not True or
            environment.get("type") != "openai_hosted" or
            (environment.get("network") or {}).get("access") != "disabled"):
        raise ProtocolError("Effective agent or network settings differ from the protocol")
    if agent.get("tools") or environment.get("plugins") or environment.get("skills"):
        raise ProtocolError("Unexpected registered tools, plugins or skills")
    if session.get("vault_ids") or environment.get("capability_directories"):
        raise ProtocolError("Unexpected external capabilities")
    if (environment.get("desktop") or {}).get("enabled"):
        raise ProtocolError("Unexpected browser provisioning")


def root_turn(turns: list[dict]):
    roots = [turn for turn in turns if turn.get("subagent_id") is None]
    if len(roots) > 1:
        raise ProtocolError("More than one root turn; refusing answer selection")
    return roots[0] if roots else None


def final_answer(items: list[dict], turn: dict) -> str:
    messages = [item for item in items if
                item.get("type") == "message" and item.get("role") == "assistant" and
                item.get("turn_id") == turn.get("id") and
                item.get("phase") == "final_answer" and item.get("status") == "completed"]
    if len(messages) != 1:
        raise ProtocolError("Expected exactly one completed root final_answer message")
    return "".join(part["text"] for part in messages[0]["content"] if part["type"] == "output_text")


def aggregate_usage(turns: list[dict]):
    if not turns or len({turn["id"] for turn in turns}) != len(turns):
        raise ProtocolError("Missing or duplicate turn accounting")
    if any(turn["status"] not in TERMINAL for turn in turns):
        return None
    usages = [turn.get("usage") for turn in turns]
    if any(not usage or any(usage.get(k) is None for k in ("input_tokens", "output_tokens")) for usage in usages):
        return None
    cached = [(usage.get("input_tokens_details") or {}).get("cached_tokens") for usage in usages]
    result = {key: sum(usage[key] for usage in usages) for key in ("input_tokens", "output_tokens")}
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    result["input_tokens_details"] = {"cached_tokens": sum(cached) if all(v is not None for v in cached) else None}
    return result


class SessionRunner:
    def __init__(self, client, directory: Path, system_message: str, poll_seconds: float = 5, cleanup_on_finish=True, guidance=None):
        self.client, self.directory = client, directory
        self.system_message = system_message
        self.guidance = guidance
        directory.mkdir(parents=True, exist_ok=True)
        self.state_path = directory / "state.json"
        self.state = read_json(self.state_path, {})
        self.poll_seconds = poll_seconds
        self.cleanup_on_finish = cleanup_on_finish

    def save(self):
        atomic_json(self.state_path, self.state)

    def _read(self, call):
        return call()

    def _pages(self, kind: str):

        resource = getattr(self.client.beta.agents.sessions, kind)
        return self._read(lambda: [as_dict(item) for item in resource.list(
            self.state["session_id"], order="asc", limit=100,
        )])

    def _child_pages(self, kind: str, subagent_id: str):
        resource = getattr(self.client.beta.agents.sessions.subagents, kind)
        return self._read(lambda: [as_dict(item) for item in resource.list(
            subagent_id, session_id=self.state["session_id"], order="asc", limit=100,
        )])

    def _history(self, turns):
        root_items = self._pages("items")
        by_turn = {t["id"]: t for t in turns}
        children = self._pages("subagents")
        child_ids = list(dict.fromkeys([s["id"] for s in children] +
                                      [t["subagent_id"] for t in turns if t.get("subagent_id")]))
        for child in child_ids:
            for child_turn in self._child_pages("turns", child):
                if child_turn.get("subagent_id") != child:
                    raise ProtocolError("Subagent turn belongs to a different agent")
                by_turn[child_turn["id"]] = child_turn
        return root_items, list(by_turn.values()), len(child_ids)

    def _cancel_once(self):
        if self.state.get("cancel_requested"):
            return
        self.state["cancel_requested"] = True
        self.save()
        try:
            self.client.beta.agents.sessions.events.create(
                self.state["session_id"], events=[{"type": "agent.session.input.cancel"}],
            )
        except Exception as exc:
            raise SessionInterrupted("Cancellation unconfirmed") from exc

    def _validate(self, session):
        try:
            validate_session(session, self.system_message)
        except ProtocolError as exc:
            self.state.update(phase="finished", failure_reason="protocol_error")
            self.save()
            atomic_json(self.directory / "outcome.json", {
                "status": "protocol_error", "answer_text": "", "error": str(exc),
                "elapsed_seconds": self.state.get("elapsed_seconds"), "usage": None,
                "session_usage": session.get("usage"), "turn_usage": [],
            })
            try:
                self._cancel_once()
            except SessionInterrupted:
                pass


            self.cleanup()
            raise

    def _remember_images(self, images):
        specs = []
        for number, image in enumerate(images, 1):
            name = image.filename(number)
            path = self.directory / "inputs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != image.data:
                    raise ProtocolError("Sample input bytes differ from saved data")
            else:
                temporary = path.with_suffix(path.suffix + ".partial")
                with temporary.open("wb") as handle:
                    handle.write(image.data)
                temporary.replace(path)
            specs.append({"name": name, "width": image.width, "height": image.height})
        return specs

    def _live_files(self, path):
        return self._read(lambda: [as_dict(f) for f in self.client.beta.agents.environments.files.list(
            self.state["environment_id"], path=path, order="asc", limit=100,
        )])


    def _prepare_inputs(self, specs):
        manifest, uploads = input_manifest(self.directory, specs)
        for _ in range(120):
            environment = self._read(lambda: as_dict(self.client.beta.agents.environments.retrieve(self.state["environment_id"])))
            status = environment.get("status")
            if status == "connected":
                break
            if status == "failed":
                raise EnvironmentSetupFailed(environment)
            if status in {"expired", "disconnected"}:
                raise SessionInterrupted(f"Input environment is {status}; inspect the original session")
            time.sleep(max(self.poll_seconds, 1))
        else:
            raise SessionInterrupted("Input provisioning is not ready")
        if self._pages("turns"):
            raise ProtocolError("A model turn started before input provisioning completed")
        for upload in uploads:
            path = upload["path"]
            self.state["pending_chunk"] = path
            self.save()
            self.client.beta.agents.environments.files.create(
                self.state["environment_id"], type="inline", path=path,
                data=encoded_part(self.directory, upload),
            )
            self.state.pop("pending_chunk", None)
            self.save()
        for _ in range(120):
            live = {f["path"]:f.get("size_bytes") for f in self._live_files("/workspace/input")}
            if any("/transfer_failed_" in p for p in live):
                raise ProtocolError("Original-byte input restoration failed; no model task was sent")
            if all(live.get("/workspace/input/" + m["name"]) == m["size"] for m in manifest):
                self.state.update(phase="ready", inputs_ready=True)
                self.save()
                return
            time.sleep(max(self.poll_seconds, 1))
        raise SessionInterrupted("Input environment is not ready")

    def _event(self, event: dict):
        sid = event.get("session_id") or (event.get("session") or {}).get("id")
        if sid:
            if self.state.get("session_id") not in {None, sid}:
                raise ProtocolError("Event belongs to a different session")
            if not self.state.get("session_id"):
                self.state.update(session_id=sid, phase="running")
                self.save()
        if event.get("session"):
            self._validate(event["session"])
        kind = event.get("type", "")
        return ((kind in {"agent.session.turn.completed", "agent.session.turn.failed", "agent.session.turn.cancelled"}
                 and (event.get("turn") or {}).get("subagent_id") is None) or
                kind in {"agent.session.failed", "agent.session.environment.failed", "error"} or
                kind.endswith(".requires_action"))

    def run(self, prompt: str, images) -> dict:
        settings = {"model": MODEL, "instructions": agent_instructions(self.system_message), "prompt": prompt,
                    "input_mode": "original_image_files", "input_transport":"inline_chunks",
                    "network": "disabled", "multi_agent": True}
        if self.guidance is not None:

            settings.update(arm="guided", guidance=self.guidance)
        existing = read_json(self.directory / "settings.json")
        if self.state and existing is None:
            raise ProtocolError("Existing session has no saved settings")
        if existing is not None and existing != settings:
            raise ProtocolError("Saved sample settings changed")
        atomic_json(self.directory / "settings.json", settings)
        outcome = read_json(self.directory / "outcome.json")
        if outcome is not None:
            if self.cleanup_on_finish:
                self.cleanup()
            if outcome["status"] == "protocol_error":
                raise ProtocolError("Saved protocol violation; stop further submissions")
            return outcome
        if self.state:
            if self.state.get('session_id') and self.state.get('input_started'):
                return self._monitor()
            raise SessionInterrupted('Existing session state requires inspection; no new request was sent')
        specs = self._remember_images(images)
        if self.state.get("phase") == "creating" and not self.state.get("session_id"):
            raise SessionInterrupted("Session creation is unresolved; preserve its saved state")
        if not self.state.get("session_id"):
            manifest, _ = input_manifest(self.directory, specs)
            request = build_hosted_request(setup_command(manifest), self.system_message)

            self.state.update(phase="creating", elapsed_seconds=0)
            self.save()
            try:
                session = as_dict(self.client.beta.agents.sessions.create(**request))
                if not session.get("id"):
                    raise SessionInterrupted("Ambiguous session creation without an ID")
                self.state.update(session_id=session["id"], environment_id=session["environment"]["id"], phase="preparing")
                self.save()
                self._validate(session)
            except (ProtocolError, SessionInterrupted):
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if not self.state.get("session_id"):
                    self.state["creation_error"] = api_error(exc)
                    self.save()
                    if status in {400, 401, 403, 404, 413, 422, 429}:
                        self.state.update(phase="rejected", error=api_error(exc))
                        self.save()
                        outcome = {"status": "api_rejected", "answer_text": "", "error": self.state["error"],
                                   "elapsed_seconds": None, "usage": None, "session_usage": None, "turn_usage": []}
                        atomic_json(self.directory / "outcome.json", outcome)
                        self.cleanup()
                        raise SessionInterrupted("API rejected the request; stop new submissions and inspect account/API access") from exc
                    raise SessionInterrupted("Session creation failed or is unresolved") from exc
                raise SessionInterrupted("Session creation interrupted") from exc
        if not self.state.get("session_id"):
            raise SessionInterrupted("Session creation closed without its ID")
        if not self.state.get("inputs_ready"):
            self._validate(self._read(lambda: as_dict(self.client.beta.agents.sessions.retrieve(self.state["session_id"]))))
            try:
                self._prepare_inputs(specs)
            except EnvironmentSetupFailed as exc:
                session = self._read(lambda: as_dict(self.client.beta.agents.sessions.retrieve(self.state["session_id"])))
                self._validate(session)
                turns = self._pages("turns")
                if turns or self.state.get("input_started"):
                    raise SessionInterrupted("Environment setup failure has model activity") from exc
                self.state.update(failure_reason="environment_failed")
                self.save()
                session["error"] = exc.environment.get("error") or session.get("error") or {
                    "code": "environment_failed", "message": str(exc),
                }


                return self._finish(session, turns, None)
            except ProtocolError as exc:
                self.state.update(phase="finished", failure_reason="input_preparation_error")
                self.save()
                atomic_json(self.directory / "outcome.json", {"status":"protocol_error", "answer_text":"", "error":str(exc),
                            "elapsed_seconds":None, "usage":None, "session_usage":None, "turn_usage":[]})
                self.cleanup()
                raise
        if not self.state.get("input_started"):


            try:
                with self._read(lambda: self.client.beta.agents.sessions.events.stream(self.state["session_id"])) as events:
                    self.state.update(input_started=True, phase="submitting")
                    self.save()
                    self.client.beta.agents.sessions.events.create(
                        self.state["session_id"], events=[{"type":"agent.session.input.message", "input":build_input(prompt, specs, self.guidance)}],
                        idempotency_key=self.state["session_id"] + ":initial-task",
                    )
                    self.state["phase"] = "running"
                    self.save()
                    for event in events:
                        if self._event(as_dict(event)):
                            break
            except (ProtocolError, SessionInterrupted):
                raise
            except Exception as exc:
                self.state["transport_error"] = api_error(exc)
                self.save()
                if not self.state.get("input_started"):
                    raise SessionInterrupted("Cannot open events") from exc
                raise SessionInterrupted("Task submission interrupted; preserve the original session") from exc
        return self._monitor()

    def _monitor(self):
        started = time.monotonic()
        prior_elapsed = self.state.get("elapsed_seconds", 0)
        empty_reads = 0
        while True:
            sid = self.state["session_id"]
            session = self._read(lambda: as_dict(self.client.beta.agents.sessions.retrieve(sid)))
            self._validate(session)
            turns = self._pages("turns")
            turn = root_turn(turns)
            self.state["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
            if session.get("created_at") is not None:
                end = (turn or {}).get("completed_at") or time.time()
                self.state["elapsed_seconds"] = max(0, end - session["created_at"])
            self.save()
            if turn is None and session.get("status") != "failed":
                empty_reads += 1
                if empty_reads >= 12:
                    raise SessionInterrupted("Task submission not confirmed by a turn")
            if turn and turn["status"] in TERMINAL:
                return self._finish(session, turns, turn)
            if session["status"] == "failed":
                return self._finish(session, turns, turn)
            if session.get("required_actions"):


                if any(a["type"] != "environment_connection" for a in session["required_actions"]):
                    self.state["failure_reason"] = "required_external_action"
                    self.save()
                    self._cancel_once()
            time.sleep(self.poll_seconds)

    def _finish(self, session, turns, turn):
        items, turns, child_count = self._history(turns)
        if session.get('status') != 'failed' and any(t.get('subagent_id') and t.get('status') not in TERMINAL for t in turns):
            raise SessionInterrupted('Subagent turns are still active')
        status = self.state.get("failure_reason") or (
            "failed" if session.get("status") == "failed" else turn["status"] if turn else "failed"
        )
        answer, error = "", (turn or {}).get("error") or session.get("error")
        if status == "completed":
            try:
                answer = final_answer(items, turn)
            except ProtocolError as exc:
                raise SessionInterrupted(str(exc)) from exc
        outcome = {
            "status": status, "answer_text": answer, "error": clean_evidence(error),
            "elapsed_seconds": self.state.get("elapsed_seconds"),
            "usage": aggregate_usage(turns) if turns else None,
            "session_usage": session.get("usage"),
            "subagent_count": child_count,
        }
        atomic_json(self.directory / "outcome.json", outcome)
        self.state["phase"] = "finished"
        self.save()
        if self.cleanup_on_finish:
            self.cleanup()
        if isinstance(error, dict) and error.get("code") in FATAL_SERVICE_CODES:
            raise SessionInterrupted("Service access/budget failure retained; stop further submissions")
        return outcome

    def cleanup(self):
        if not (self.directory / "outcome.json").exists() or self.state.get("cleaned"):
            return
        try:
            if self.state.get("session_id") and not self.state.get("session_deleted"):
                self._delete(lambda: self.client.beta.agents.sessions.delete(self.state["session_id"]))
                self.state["session_deleted"] = True
                self.save()
            deleted = self.state.setdefault("deleted_files", [])
            for file_id in self.state.get("uploaded_files", {}).values():
                if file_id not in deleted:
                    self._delete(lambda: self.client.files.delete(file_id))
                    deleted.append(file_id)
                    self.save()
            self.state["cleaned"] = True
            self.state.pop("cleanup_error", None)
            self.save()
        except Exception as exc:
            self.state["cleanup_error"] = api_error(exc)
            self.save()
            raise SessionInterrupted("Prediction saved; resource cleanup failed") from exc

    def _delete(self, call):
        try:
            return call()
        except Exception as exc:
            if getattr(exc, 'status_code', None) != 404:
                raise



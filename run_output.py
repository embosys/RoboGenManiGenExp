"""Scene-local run records for the external-scene adapter."""
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def ray_temp_dir(run_dir):
    run_path = str(Path(run_dir).expanduser().resolve()).encode('utf-8')
    return str(Path('/tmp') / ('rg-' + hashlib.sha256(run_path).hexdigest()[:12]))


@contextmanager
def capture_console(path):
    # ponytail: Python streams only; native simulator output remains in the terminal.
    class Tee:
        def __init__(self, console, log):
            self.console, self.log = console, log

        def write(self, text):
            self.console.write(text)
            # Logging handlers can retain this stream after capture has ended.
            if not self.log.closed:
                self.log.write(text)
                self.log.flush()
            return len(text)

        def flush(self):
            self.console.flush()
            if not self.log.closed:
                self.log.flush()

        def fileno(self):
            return self.console.fileno()

        def close(self):
            # The context owns the log; the caller owns the console stream.
            self.flush()

        def __getattr__(self, name):
            return getattr(self.console, name)

    with open(path, 'a', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


class RunOutput:
    def __init__(self, args, scene):
        self.phase = 'initialization'
        self.started = time.monotonic()
        root = args.output_dir.expanduser().resolve() if args.output_dir else scene.parent / 'robogen'
        batch_identity = {key: os.environ[env] for env, key in (
            ('MANIGEN_BATCH_HASH', 'batch_hash'),
            ('MANIGEN_BATCH_RUN_ID', 'batch_run_id'),
            ('MANIGEN_BATCH_NAME', 'batch_name'),
        ) if env in os.environ}
        if args.run_dir:
            self.path = args.run_dir.expanduser().resolve()
            self.meta = json.loads((self.path / 'run_meta.json').read_text())
            if self.meta['scene'] != str(scene) or self.meta['scene_sha256'] != hashlib.sha256(scene.read_bytes()).hexdigest():
                raise ValueError('Run directory belongs to another scene or the scene has changed')
            for key, value in batch_identity.items():
                if self.meta.get(key) != value:
                    raise ValueError('Run directory has a different ' + key)
            for name in ('seed', 'robot_mode'):
                value = getattr(args, name)
                if value is not None and value != self.meta[name]:
                    raise ValueError('Run directory has a different ' + name)
                setattr(args, name, self.meta[name])
        else:
            args.seed = args.seed if args.seed is not None else 0
            args.robot_mode = args.robot_mode or 'robogen'
            run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S') + '_s%03d' % args.seed
            if run_id in ('.', '..', '_assets') or Path(run_id).name != run_id or '\\' in run_id:
                raise ValueError('--run-id must be a single directory name')
            for index in range(1000):
                name = run_id if index == 0 else run_id + '-%02d' % index
                self.path = root / name
                try:
                    self.path.mkdir(parents=True, exist_ok=False)
                    break
                except FileExistsError:
                    continue
            else:
                raise ValueError('Too many duplicate run IDs')
            self.meta = {'run_id': self.path.name, 'method': 'robogen', 'scene': str(scene),
                         'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
                         'robot_mode': args.robot_mode, 'seed': args.seed, 'training_seed': 0,
                         'started_at': datetime.now(timezone.utc).isoformat(), 'sessions': []}
            self.meta.update(batch_identity)
            try:
                repository = Path(__file__).resolve().parent
                self.meta['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, stderr=subprocess.DEVNULL, text=True).strip()
                self.meta['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repository, stderr=subprocess.DEVNULL, text=True).strip())
            except (OSError, subprocess.CalledProcessError):
                self.meta['source_commit'] = None
        for name in ('llm', 'code', 'trajectory', 'training', 'artifacts'):
            (self.path / name).mkdir(exist_ok=True)
        # Each continued command is a session; earlier result snapshots are retained.
        self.result = {'status': 'running', 'native_success': None, 'success': None,
                       'failure_stage': None, 'failure_reason': None, 'attempts': []}
        self.session = {'action': args.action, 'started_at': datetime.now(timezone.utc).isoformat(),
                        'args': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
        self.meta['sessions'].append(self.session)
        self.persist()

    def persist(self):
        write_json(self.path / 'run_meta.json', self.meta)
        write_json(self.path / 'result.json', self.result)

    def finish(self, error=None):
        self.session.update(finished_at=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=time.monotonic() - self.started,
                            status='error' if error is not None else 'completed')
        self.result.update(status=self.session['status'], elapsed_seconds=self.session['elapsed_seconds'],
                           action=self.session['action'])
        if error is not None:
            self.result.update(failure_stage=self.phase, failure_reason=str(error))
        calls = self.path / 'llm/index.json'
        self.result['llm_call_count'] = len(json.loads(calls.read_text())) if calls.exists() else 0
        self.session['result'] = self.result.copy()
        self.persist()

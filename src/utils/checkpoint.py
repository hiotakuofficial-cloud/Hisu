"""Checkpoint management"""

import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional


class CheckpointManager:
    """Manage model checkpoints"""

    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, state: Dict[str, Any], filename: str, metadata: Optional[Dict] = None):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint_data = {
            'state': state,
            'metadata': metadata or {}
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        if metadata:
            metadata_path = checkpoint_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {filename} not found")

        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        return checkpoint_data

    def list_checkpoints(self) -> list:
        """List all checkpoints"""
        return [f.name for f in self.checkpoint_dir.glob('*.pkl')]

    def delete_checkpoint(self, filename: str):
        """Delete checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            metadata_path.unlink()

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('*.pkl'))

        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest.name

    def cleanup_old_checkpoints(self, keep_n: int = 5):
        """Keep only the n most recent checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('*.pkl'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep_n:]:
            checkpoint.unlink()

            metadata_path = checkpoint.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()

from __future__ import annotations

import os
from typing import List, Union
from tdw.controller import Controller
from requests import get
from requests.exceptions import RequestException
from pathlib import Path
import shutil
import hashlib

class AssetCachedController(Controller):
    def __init__(self, cache_dir="transport_challenge_asset_bundles", **kwargs):
        self.cache_dir = None
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            self.cache_dir = Path(cache_dir)
        super().__init__(**kwargs)
    
    def communicate(self, commands: dict | List[dict]):
        '''
        override and calculate add_ons' commands in advance to make cache of asset
        '''
        if self.cache_dir is None:
            return super().communicate(commands)
        
        if isinstance(commands, dict):
            commands = [commands]
        add_ons_t = self.add_ons
        self.add_ons = [] # remove add_ons
        for m in add_ons_t:
            if not m.initialized:
                commands.extend(m.get_initialization_commands())
                m.initialized = True
            else:
                commands.extend(m.commands)
                m.commands.clear()
        for m in add_ons_t:
            m.before_send(commands)
        
        for cmd in commands:
            if "url" in cmd:
                cmd["url"] = self.get_asset(cmd["url"])
        resp = super().communicate(commands)
        for m in add_ons_t:
            m.on_send(resp)
        self.add_ons = add_ons_t # resume add_ons
        return resp
    
    def _is_valid_asset_bundle(self, path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < 16:
                return False
            with path.open("rb") as f:
                head = f.read(7)
            return head in (b"UnityFS", b"UnityRaw", b"UnityWeb")
        except OSError:
            return False

    def get_asset(self, url: str) -> str:
        name = hashlib.md5(url.encode()).hexdigest()
        path = self.cache_dir.joinpath(name)
        resolved = path.resolve()

        if not self._is_valid_asset_bundle(resolved):
            try:
                print(f"downloading {url} to {str(resolved)}")
                resp = get(url, timeout=60)
                resp.raise_for_status()
                content = resp.content
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(content)
            except (RequestException, OSError) as e:
                try:
                    if resolved.exists():
                        resolved.unlink()
                except OSError:
                    pass
                raise RuntimeError(f"Failed to download asset bundle: {url}") from e

            if not self._is_valid_asset_bundle(resolved):
                try:
                    resolved.unlink()
                except OSError:
                    pass
                raise RuntimeError(f"Downloaded file is not a valid Unity asset bundle: {url}")

        return resolved.as_uri()
    
    def clear_cache(self):
        shutil.rmtree(self.cache_dir.resolve())
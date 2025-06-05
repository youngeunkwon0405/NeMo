# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, Tuple
from urllib.parse import urlparse

try:
    from nemo import __version__ as NEMO_VERSION
except ImportError:
    NEMO_VERSION = 'git'

from nemo import constants
from nemo.utils import logging
from nemo.utils.nemo_logging import LogMode

try:
    from lhotse.serialization import open_best as lhotse_open_best

    LHOTSE_AVAILABLE = True
except ImportError:
    LHOTSE_AVAILABLE = False


def resolve_cache_dir() -> pathlib.Path:
    """
    Utility method to resolve a cache directory for NeMo that can be overriden by an environment variable.

    Example:
        NEMO_CACHE_DIR="~/nemo_cache_dir/" python nemo_example_script.py

    Returns:
        A Path object, resolved to the absolute path of the cache directory. If no override is provided,
        uses an inbuilt default which adapts to nemo versions strings.
    """
    override_dir = os.environ.get(constants.NEMO_ENV_CACHE_DIR, "")
    if override_dir == "":
        path = pathlib.Path.joinpath(pathlib.Path.home(), f'.cache/torch/NeMo/NeMo_{NEMO_VERSION}')
    else:
        path = pathlib.Path(override_dir).resolve()
    return path


def is_datastore_path(path) -> bool:
    """Check if a path is from a data object store."""
    try:
        result = urlparse(path)
        return bool(result.scheme) and bool(result.netloc)
    except AttributeError:
        return False


def is_tarred_path(path) -> bool:
    """Check if a path is for a tarred file."""
    return path.endswith('.tar')


def is_datastore_cache_shared() -> bool:
    """Check if store cache is shared."""
    # Assume cache is shared by default, e.g., as in resolve_cache_dir (~/.cache)
    cache_shared = int(os.environ.get(constants.NEMO_ENV_DATA_STORE_CACHE_SHARED, 1))

    if cache_shared == 0:
        return False
    elif cache_shared == 1:
        return True
    else:
        raise ValueError(f'Unexpected value of env {constants.NEMO_ENV_DATA_STORE_CACHE_SHARED}')


def ais_cache_base() -> str:
    """Return path to local cache for AIS."""
    override_dir = os.environ.get(constants.NEMO_ENV_DATA_STORE_CACHE_DIR, "")
    if override_dir == "":
        cache_dir = resolve_cache_dir().as_posix()
    else:
        cache_dir = pathlib.Path(override_dir).resolve().as_posix()

    if cache_dir.endswith(NEMO_VERSION):
        # Prevent re-caching dataset after upgrading NeMo
        cache_dir = os.path.dirname(cache_dir)
    return os.path.join(cache_dir, 'ais')


def ais_endpoint() -> str:
    """Get configured AIS endpoint."""
    return os.getenv('AIS_ENDPOINT')


def bucket_and_object_from_uri(uri: str) -> Tuple[str, str]:
    """Parse a path to determine bucket and object path.

    Args:
        uri: Full path to an object on an object store

    Returns:
        Tuple of strings (bucket_name, object_path)
    """
    if not is_datastore_path(uri):
        raise ValueError(f'Provided URI is not a valid store path: {uri}')
    uri_parts = pathlib.PurePath(uri).parts
    bucket = uri_parts[1]
    object_path = pathlib.PurePath(*uri_parts[2:])

    return str(bucket), str(object_path)


def ais_endpoint_to_dir(endpoint: str) -> str:
    """Convert AIS endpoint to a valid dir name.
    Used to build cache location.

    Args:
        endpoint: AIStore endpoint in format https://host:port

    Returns:
        Directory formed as `host/port`.
    """
    result = urlparse(endpoint)
    if not result.hostname or not result.port:
        raise ValueError(f"Unexpected format for ais endpoint: {endpoint}")
    return os.path.join(result.hostname, str(result.port))


@lru_cache(maxsize=1)
def ais_binary() -> str:
    """Return location of `ais` binary if available."""
    path = shutil.which('ais')

    if path is not None:
        logging.debug('Found AIS binary at %s', path)
        return path

    # Double-check if it exists at the default path
    default_path = '/usr/local/bin/ais'
    if os.path.isfile(default_path):
        logging.info('ais available at the default path: %s', default_path, mode=LogMode.ONCE)
        return default_path
    else:
        logging.warning(
            f'AIS binary not found with `which ais` and at the default path {default_path}.', mode=LogMode.ONCE
        )
        return None


def datastore_path_to_local_path(store_path: str) -> str:
    """Convert a data store path to a path in a local cache.

    Args:
        store_path: a path to an object on an object store

    Returns:
        Path to the same object in local cache.
    """
    if is_datastore_path(store_path):
        endpoint = ais_endpoint()
        if not endpoint:
            raise RuntimeError(f'AIS endpoint not set, cannot resolve {store_path}')

        local_ais_cache = os.path.join(ais_cache_base(), ais_endpoint_to_dir(endpoint))
        store_bucket, store_object = bucket_and_object_from_uri(store_path)
        local_path = os.path.join(local_ais_cache, store_bucket, store_object)
    else:
        raise ValueError(f'Unexpected store path format: {store_path}')

    return local_path


def open_datastore_object_with_binary(path: str, num_retries: int = 5):
    """Open a datastore object and return a file-like object.

    Args:
        path: path to an object
        num_retries: number of retries if the get command fails with ais binary, as AIS Python SDK has its own retry mechanism

    Returns:
        File-like object that supports read()
    """

    if is_datastore_path(path):
        endpoint = ais_endpoint()
        if endpoint is None:
            raise RuntimeError(f'AIS endpoint not set, cannot resolve {path}')

        binary = ais_binary()

        if not binary:
            raise RuntimeError(
                f"AIS binary is not found, cannot resolve {path}. Please either install it or install Lhotse with `pip install lhotse`.\n"
                "Lhotse's native open_best supports AIS Python SDK, which is the recommended way to operate with the data from AIStore.\n"
                "See AIS binary installation instructions at https://github.com/NVIDIA/aistore?tab=readme-ov-file#install-from-release-binaries.\n"
            )

        cmd = f'{binary} get {path} -'

        done = False

        for _ in range(num_retries):
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False  # bytes mode
            )
            stream = proc.stdout
            if stream.peek(1):
                done = True
                break

        if not done:
            error = proc.stderr.read().decode("utf-8", errors="ignore").strip()
            raise ValueError(
                f"{path} couldn't be opened with AIS binary after {num_retries} attempts because of the following exception: {error}"
            )

        return stream


def open_best(path: str, mode: str = "rb"):
    if LHOTSE_AVAILABLE:
        return lhotse_open_best(path, mode=mode)
    if is_datastore_path(path):
        return open_datastore_object_with_binary(path)
    return open(path, mode=mode)


def get_datastore_object(path: str, force: bool = False, num_retries: int = 5) -> str:
    """Download an object from a store path and return the local path.
    If the input `path` is a local path, then nothing will be done, and
    the original path will be returned.

    Args:
        path: path to an object
        force: force download, even if a local file exists
        num_retries: number of retries if the get command fails with ais binary, as AIS Python SDK has its own retry mechanism

    Returns:
        Local path of the object.
    """
    if is_datastore_path(path):

        local_path = datastore_path_to_local_path(store_path=path)

        if not os.path.isfile(local_path) or force:
            # Either we don't have the file in cache or we force download it
            # Enhancement: if local file is present, check some tag and compare against remote
            local_dir = os.path.dirname(local_path)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            with open(local_path, 'wb') as f:
                f.write(open_best(path).read(), num_retries=num_retries)

        return local_path

    else:
        # Assume the file is local
        return path


class DataStoreObject:
    """A simple class for handling objects in a data store.
    Currently, this class supports objects on AIStore.

    Args:
        store_path: path to a store object
        local_path: path to a local object, may be used to upload local object to store
        get: get the object from a store
    """

    def __init__(self, store_path: str, local_path: str = None, get: bool = False):
        if local_path is not None:
            raise NotImplementedError('Specifying a local path is currently not supported.')

        self._store_path = store_path
        self._local_path = local_path

        if get:
            self.get()

    @property
    def store_path(self) -> str:
        """Return store path of the object."""
        return self._store_path

    @property
    def local_path(self) -> str:
        """Return local path of the object."""
        return self._local_path

    def get(self, force: bool = False) -> str:
        """Get an object from the store to local cache and return the local path.

        Args:
            force: force download, even if a local file exists

        Returns:
            Path to a local object.
        """
        if not self.local_path:
            # Assume the object needs to be downloaded
            self._local_path = get_datastore_object(self.store_path, force=force)
        return self.local_path

    def put(self, force: bool = False) -> str:
        """Push to remote and return the store path

        Args:
            force: force download, even if a local file exists

        Returns:
            Path to a (remote) object object on the object store.
        """
        raise NotImplementedError()

    def __str__(self):
        """Return a human-readable description of the object."""
        description = f'{type(self)}: store_path={self.store_path}, local_path={self.local_path}'
        return description


def datastore_object_get(store_object: DataStoreObject) -> bool:
    """A convenience wrapper for multiprocessing.imap.

    Args:
        store_object: An instance of DataStoreObject

    Returns:
        True if get() returned a path.
    """
    return store_object.get() is not None


def wds_url_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool],
    **kw: Dict[str, Any],
):
    """
    Open URLs and yield a stream of url+stream pairs.
    This is a workaround to use lhotse's open_best instead of webdataset's default url_opener.
    webdataset's default url_opener uses gopen, which does not support opening datastore paths.

    Args:
        data: Iterator over dict(url=...).
        handler: Exception handler.
        **kw: Keyword arguments for gopen.gopen.

    Yields:
        A stream of url+stream pairs.
    """
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = open_best(url, mode="rb")
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break

from typing import NamedTuple
import logging
import torch
from pymongo import MongoClient
import gridfs
import hashlib
import io
from compress_utils import compress_buffer

logger = logging.getLogger(__name__)


class StorageConfig(NamedTuple):
    hostname: str
    port: int
    username: str
    password: str


class Storage:
    def __init__(self, config: StorageConfig, reset: bool = False):
        try:
            self.client = MongoClient(
                host=config.hostname,
                port=config.port,
                username=config.username,
                password=config.password,
            )
        except ConnectionError as e:
            logger.exception(e)

        self.latest_online_id = ""
        self.latest_target_id = ""

        print("storage connected.")
        if reset:
            db = self.client["model_weights"]
            db.drop_collection(db["ids"])
            db.drop_collection(db["fs.files"])
            print("dropped existing collections")
        else:
            self.latest_online_id = self.get_latest_weights_id("online_weights")
            self.latest_target_id = self.get_latest_weights_id("target_weights")

    def get_latest_weights_id(self, key):
        db = self.client["model_weights"]
        ids = db["ids"]
        id = ids.find_one({key: {"$exists": True}})

        if id:
            return id[key]
        return None

    def update_weights_id(self, key: str, id: str):
        db = self.client["model_weights"]
        ids = db["ids"]
        ids.update_one(
            filter={{key: {"$exists": True}}},
            update={"$set": {key: id}},
            upsert=True,
        )

    def _store(self, model: torch.nn.Module, id_key: str) -> str:
        db = self.client["model_weights"]
        fs = gridfs.GridFS(db)

        with io.BytesIO() as buffer:
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            compressed = compress_buffer(buffer)
            logger.info(f"{id_key} hash: {hashlib.md5(compressed).hexdigest()}")
            id = fs.put(compressed)

        prev_id = self.get_latest_weights_id(id_key)
        if not prev_id is None:
            fs.delete(prev_id)

        self.update_weights_id(id_key, id)

    def store_models(self, online_model: torch.nn.Module, target_model: torch.nn.Module):
        self._store(online_model, "online_weights")
        self._store(target_model, "target_weights")

    def get_weights(self):
        while True:
            try:
                online_id = self.get_latest_weights_id("online_weights")
                target_id = self.get_latest_weights_id("target_weights")
                logger.info(f"weights ids: {online_id}, {target_id}")

                if (
                    (not online_id)
                    or (not target_id)
                    or online_id == self.latest_online_id
                    or target_id == self.latest_target_id
                ):
                    return None, None

                self.latest_online_id = online_id
                self.latest_target_id = target_id

                db = self.client["model_weights"]
                fs = gridfs.GridFS(db)

                online_weights = fs.get(online_id).read()
                target_weights = fs.get(target_id).read()
                logger.info(f"online hash: {hashlib.md5(online_weights).hexdigest()}")
                logger.info(f"target hash: {hashlib.md5(target_weights).hexdigest()}")
                return online_weights, target_weights
            except Exception as e:
                logger.warning("error getting weights: ", e)

from typing import NamedTuple
import time
import logging
from pymongo import MongoClient
import gridfs

logger = logging.getLogger(__name__)


class StorageConfig(NamedTuple):
    hostname: str
    port: int
    username: str
    password: str


class Storage:
    def __init__(self, config: StorageConfig, reset: bool = False):
        self.client = MongoClient(
            host=config.hostname,
            port=config.port,
            username=config.username,
            password=config.password,
        )

        if reset:
            db = self.client["model_weights"]
            db.drop_collection(db["ids"])
            db.drop_collection(db["fs.files"])

    def get_latest_weights_id(self):
        db = self.client["model_weights"]
        ids = db["ids"]
        id = ids.find_one()

        if id:
            return id["weights_id"]
        return None

    def update_weights_id(self, id):
        db = self.client["model_weights"]
        ids = db["ids"]
        ids.update_one(
            filter={},
            update={"$set": {"weights_id": id}},
            upsert=True,
        )

    def store_weights(self, bytes):
        db = self.client["model_weights"]
        fs = gridfs.GridFS(db)
        id = fs.put(bytes)

        # delete previous weights if they exist
        prev_id = self.get_latest_weights_id()
        if prev_id is not None:
            fs.delete(self.get_latest_weights_id())

        # update the latest weights id
        self.update_weights_id(id)

    def get_weights(self):
        id = self.get_latest_weights_id()
        if not id:
            return None

        db = self.client["model_weights"]
        fs = gridfs.GridFS(db)

        return fs.get(id).read()

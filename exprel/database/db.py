from tinydb import TinyDB, Query
import os


class Database:
    def __init__(self, path_to_db=None):
        if path_to_db == None:
            self.path = f"{os.path.dirname(__file__)}/db.json"
        else:
            self.path = path_to_db
        self.db = TinyDB(self.path)

    def insert_document(self, sen_id, document):
        self.db.insert({"doc_id": sen_id, "processed": document.to_dict()})

    def query_document(self, sen_id):
        documents = self.db.get(doc_id=sen_id)

        return documents

    def get_all(self):
        return self.db.all()

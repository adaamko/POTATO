from exprel.dataset.sample import Sample
from bs4 import BeautifulSoup


class SemevalSample(Sample):
    def __init__(self, sen_id, sentence, label, nlp, db):
        super().__init__()
        self.sen_id = sen_id
        self.e1 = None
        self.e2 = None
        self.prepare_sentence(sentence)
        self.label = label
        self.nlp = nlp
        self.db = db
        #self.prepare_doc()

    @property
    def sentence(self):
        return self._sentence

    @sentence.setter
    def sentence(self, sen):
        self.prepare_sentence(sen)

    def prepare_sentence(self, sen):
        soup = BeautifulSoup(sen)
        self._sentence = soup.text.strip('"')
        self.e1 = soup.e1.text
        self.e2 = soup.e2.text

    def prepare_doc(self):
        documents = self.db.query_document(self.sen_id)
        if not documents:
            doc = self.nlp(self._sentence)
            self.db.insert_document(self.sen_id, doc)

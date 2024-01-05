# larch
LLM toolbox

The tool tentatively has the following components to create any downstream llm-based search engine.

Installation can be don through:
```bash
# direct remote installation
pip install git+ssh://git@github.com/NASA-IMPACT/larch.git

# or clone and install
git clone -b develop git@github.com:NASA-IMPACT/larch.git
cd larch
pip install -e # or python setup.py install
```


# Components

## 1. DocumentIndexer

`larch.indexing.DocumentIndexer` allows to query (`.query(..., top_k=)` and fetch top_k (`.query_top_k(...)`) documents that are indexed in a the document store.
- `larch.indexing.SinequaDocumentIndexer` is used to directly connect to Sinequa to fetch relevant documents (no indexing method is allowed in this as it's left to sinequa itself)
- `larch.indexing.PaperQADocumentIndexer` uses paperqa to index documents (`index_documents(<paths>)`)
- `larch.indexing.LangchainDocumentIndexer` allows to switch between any vector store (faiss, pgvector, etc) to allow for indexing. (`index_documents(<paths>)`)
- `larch.indexing.DocumentMetadataIndexer` allows to extract/index/dump metadata extracted from files which are then further used in downstream SQL agent. This takes in `larch.metadata.MetadataExtractor` and pydantic schema applied to each document.

## 2. MetadataExtractor

`larch.metadata._base.AbstractMetadataExtractor` allows for implementing any downstream metadata extractor.

- `larch.metadata.extractors_openai.InstructorBasedOpenAIMetadataExtractor` is the standard extractor that is recommended to use. This uses function-calling. *
- `larch.metadata.extractors.LangchainBasedMetadataExtractor` uses vanilla langchain and prompting to extract metadata.
- `larch.metadata.extractors.LegacyMetadataExtractor` is a refactored older algorithm from IMPACT.


## 3. SearchEngine

`larch.search.engines.AbstractSearchEngine` component is used to abstract query-to-response process to generate answer for given user query. All the downstream search engine has to implement `query(query=<str>, top_k=<top_k>)` method.

- `larch.search.engines.SimpleRAG` simply uses any `DocumentIndexer` (especially the `query(...)` method) to wrap a RAG pipeline. (Alternatively, one can always use `.query(...)` method from the document indexer).
- `larch.search.engines.InMemoryDocumentQAEngine` takes in N documents on top of which QA can be done. This can be used standalone engine as well as can be used with `DocumentStoreRAG`
- `larch.search.engines.DocumentStoreRAG` uses the `InMemoryDocumentQAEngine` and `DocumentIndexer` for QA. `DocumentIndexer.query_top_k(...)` is used to fetch top k relevant documents which are then fed to the QA engine. *
- `larch.search.engines.SQLAgentSearchEngine` connects to a given database (and set of tables in the database), generates SQL query for a given query and generates the response for the query by fetching relevant rows from the database. This is used only if we want more complex tasks like aggregation, analysis and recommended to be used only if `MetadataExtractor` performs accurately. *
- `larch.search.engines.MultiRetrieverSearchEngine` takes in an arbitrary number of `AbstractSearchEngine`(retrievers/sources) and generates individual responses for a given query from each retriever and finally consolidates the responses from them. This is the recommended way to ensemble multiple engine retriever in larch. (Note, each retriever is run in parallel).
- `larch.search.engines.EnsembleAugmentedSearchEngine` is a very naive engine that takes in multiple engines, runs through each of them sequentially and puts all the responses in single context prompt and uses LLM to do the QA. Not recommended for now.

## 4. MetadataValidator

`larch.metadata.MetadataValidator` is used to post-process the extracted metadata.
- `larch.metadata.validators.SimpleInTextMetadataValidator` checks if the extracted value of a field in the metadata lies in the text. If it doesn't, that field is removed. (Not recommended to use)
- `larch.metadata.validators.WhitelistBasedMetadataValidator` uses a whitelist to standardize the extracted value in a field. Each field value could have set of alternate values. Fuzzy-matching is used to figure out whether to standardize or not.

## 5. MetadataEvaluator

`larch.metadata.MetadataEvaluator` is used to evaluate (numerically) the extraction of metadata.

- `larch.metadata.evaluators.JaccardEvaluator` computes the ratio of tokens found between prediction and reference (doesn't account for word ordering)
- `larch.metadata.evaluators.FlattenedExactMatcher` computes the score by flattening the prediction and reference metadata and comparing the values. (better than `JaccardEvaluator`)
- `larch.metadata.evaluators.RecursiveFuzzyMatcher` is the recommended evaluator that performs weighted scoring for each node. (See documentation for more)*

## 6. TextProcessor

`larch.processors.TextProcessor` allows for processing text. All the text processors takes in a text and egest out processed text.

- `larch.processors.PIIRemover` uses spacy to identify Personal Identification Information (name, email, phone number) and mask them out.
- `larch.processors.NonAlphaNumericRemover` removes non-alpha-numeric characters from the text
- `larch.processors.TextProcessingPipeline` is a container to hold all the text processors and run them sequentially.

---

# Usage

We can do:
- metadata extraction
- index documents into vector store
- json dump metadata in bulk
- create RAG pipeline
- etc

## Metadata Extraction

Extract from single document text

```python
from larch.metadata import InstructorBasedOpenAIMetadataExtractor
from larch.metadata.validators import WhitelistBasedMetadataValidator
from larch.processors import PIIRemover, TextProcessingPipeline
from larch.utils import load_whitelist

text_processor = TextProcessingPipeline(
    lambda x: re.sub(r"\$(?=\w|\n|\()", " ", x).strip(),
    lambda x: re.sub(r"\)(?=\w|\n|\()", " ", x).strip(),
    lambda x: re.sub(r"\#(?=\w|\n|\()", " ", x).strip(),
    lambda x: x.replace("\t", " ").replace("!", " ").strip(),
    PIIRemover()
)

schema = <pydantic schema>
whitelists = load_whitelist(<path_to_excel>)

metadata_extractor = InstructorBasedOpenAIMetadataExtractor(
    model="gpt-4",
    schema=schema,
    preprocessor=text_processor,
    debug=True,
)
validator = WhitelistBasedMetadataValidator(whitelists=whitelists, fuzzy_threshold=0.95, ...)

text = <document text>

metadata = metadata_extractor(text)
metadata = validator(metadata)
```

Extract in bulk and json dump

```python
from larch.indexing import DocumentMetadataIndexer

metadata_indexer = DocumentMetadataIndexer(
    schema,
    metadata_extractor = metadata_extractor,
    skip_errors=True,
    text_preprocessor=text_processor,
    debug=True,
)

file_paths = <paths>

# start indexing
metadata_indexer.index_documents(paths=file_paths, save_path=<path_to_json_file>)

# load existing indices
metadata_indexer = metadata_indexer.load_index(<path_to_json_file>)

# access the metadata store dict
metadata_indexer.metadata_store
```

## Document Indexing

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter

from larch.indexing import PaperQADocumentIndexer, LangchainDocumentIndexer

model = "gpt-3.5-turbo-0613"
embedder = OpenAIEmbeddings()

text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

vector_store = PGVector(
    collection_name="test_collection",
    connection_string="postgresql://...",
    embedding_function=embedder,
)

# if vector_store is None, FAISS is used by default
document_indexer = LangchainDocumentIndexer(
    llm=ChatOpenAI(model=model, temperature=0.0),
    text_preprocessor=text_processor,
    vector_store=vector_store,
     # vector_store=FAISS.load_local("../tmp/vectorstore", embeddings=embedder, index_name="test_index"),
    text_splitter=text_splitter,
    debug=True,
)

# get number of chunks in the store
print(document_indexer.num_chunks)

# or use paperqa
document_indexer = PaperQADocumentIndexer(
    llm=ChatOpenAI(model=model, temperature=0.0),
    text_preprocessor=text_processor,
    debug=True,
    name="test",
)#.load_index(<path_to_pickle>)

# get files that are indexed
print(document_indexer.docs)
```

## search engine


```python
from larch.search.engines import InMemoryDocumentQAEngine, SQLAgentSearchEngine, MultiRetrieverSearchEngine

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.0,
)

top_k = 5

engines = [
    SinequaDocumentIndexer(
        base_url=os.environ.get("SINEQUA_BASE_URL"),
        auth_token=os.environ.get("SINEQUA_ACCESS_TOKEN")
    ),
    LangchainDocumentIndexer(...),
    SQLAgentSearchEngine(
        llm=llm,
        db_uri=<db_uri>,
        tables=None, # or provide a list of table names
        debug=True,
        prompt_prefix=False,
        query_augmentation_prompt=<prompt_suffix>,
        sql_fuzzy_threshold=0.75,
        railguard_response=True,
    )
]

# create multi-retriever engine
search_engine = MultiRetrieverSearchEngine(*engines, llm=llm)

query = <query_text>
response = search_engine(query, top_k=top_k)

# we can also use individual engine which has same interface
search_engine = engines[1]
response = search_engine(query, top_k=top_k)
```

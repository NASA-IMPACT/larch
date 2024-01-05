## Using the Extractors
### Chunk-Based Metadata Extraction
The chunk-based extraction breaks input texts into chunks by tokens, then passes each chunk in parallel to open AI before aggregating the multiple resulting metadata extractions into a consolidated metadata output.

```
from larch.metadata.chunker import TokenChunker, InstructorAggregator
from larch.metadata.extractors import ChunkBasedMetadataExtractor, InstructorBasedOpenAIMetadataExtractor
from larch.schema import Metadata

with open("assesment_text.txt", "r") as file:
    text = file.read()

extractor = InstructorBasedOpenAIMetadataExtractor(
    model="gpt-3.5-turbo-0613",
    schema=Metadata,
)
chunker = TokenChunker(chunk_size=1000, chunk_overlap=5)
aggregator = InstructorAggregator(schema=Metadata)
parallel_extractor = ChunkBasedMetadataExtractor(extractor, chunker, aggregator, n_jobs=4)

parallel_extractor(text)
```

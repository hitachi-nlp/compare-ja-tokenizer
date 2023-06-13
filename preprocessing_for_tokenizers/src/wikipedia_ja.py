import datasets
import tensorflow_datasets as tfds


class WikiConfig(datasets.BuilderConfig):
    """BuilderConfig."""

    def __init__(self, **kwargs):
        """BuilderConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiConfig, self).__init__(**kwargs)


class WikipediaJA(datasets.GeneratorBasedBuilder):
    """Wikipedia Japanese dump dataset builder."""

    BUILDER_CONFIGS = [
        WikiConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Wikipedia dump data in Japanese.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Wikipedia Japanese",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "title": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )
    
    def _split_generators(self, kwargs):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN),
        ]

    def _generate_examples(self):
        """Generate wiki samples.  
        
        Reference: 
            https://huggingface.co/docs/datasets/dataset_script#generate-samples
        """
        wiki_dataset = tfds.load(name='wikipedia/20201201.ja', split='train')
        key = 0
        for wiki in wiki_dataset.as_numpy_iterator():
            text = wiki['text'].decode()
            title = wiki['title'].decode()
            yield key, {"text": text, "title": title}
            key += 1

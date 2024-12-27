from lhotse.recipes.fleurs import prepare_fleurs

langs = [
    "ar_eg",
    "en_us",
    "es_419",
    "fr_fr",
    "pt_br",
    "ru_ru",
]

manifests = prepare_fleurs(
    corpus_dir='/export/common/data/corpora/fleurs',
    output_dir='/exp/ddegenaro/fleurs',
    languages=langs,
    num_jobs=1
)

print(manifests)

for manifest in manifests:
    print(manifest)
    break
mkdir model/
curl -L https://huggingface.co/google/owlvit-base-patch32/resolve/main/pytorch_model.bin?download=true --output model/pytorch_model.bin
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/config.json --output model/config.json
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/preprocessor_config.json --output model/preprocessor_config.json
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/vocab.json --output model/vocab.json
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/tokenizer_config.json --output model/tokenizer_config.json
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/special_tokens_map.json --output model/special_tokens_map.json
curl -L https://huggingface.co/google/owlvit-base-patch32/raw/main/merges.txt --output model/merges.txt

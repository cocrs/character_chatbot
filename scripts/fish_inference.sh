python fish-speech/tools/vqgan/inference.py \
    -i voices/cut.wav \
    --output-path "output/fake.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.2-sft/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
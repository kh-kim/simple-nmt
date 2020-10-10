MODEL_FN=$1
GPU_ID=$2
BEAM_SIZE=5

VALID_FN=./corpus.shuf.valid.tok.bpe.head-1000.en
REF_VALID_FN=./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko

TEST_FN=./corpus.shuf.test.tok.bpe.head-1000.en
REF_TEST_FN=./corpus.shuf.test.tok.bpe.head-1000.detok.tok.ko

cat ${VALID_FN} | python ../translate.py --model ${MODEL_FN} --gpu_id ${GPU_ID} --lang enko --beam_size ${BEAM_SIZE} | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ${REF_VALID_FN}

cat ${TEST_FN} | python ../translate.py --model ${MODEL_FN} --gpu_id ${GPU_ID} --lang enko --beam_size ${BEAM_SIZE} | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ${REF_TEST_FN}

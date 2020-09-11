curl -L http://www.openslr.org/resources/12/dev-clean.tar.gz | tar xzv
curl -L http://www.openslr.org/resources/12/test-clean.tar.gz | tar zxv 
curl -L http://www.openslr.org/resources/12/train-clean-100.tar.gz | tar zxv
BASE_DIR=`dirname $0`
python  ${BASE_DIR}/wav2vec_manifest.py ./LibriSpeech --dest $1

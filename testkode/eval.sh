case $1 in
    "reg") python ../../EmoInt/evaluate.py 4 ../../kode/testkode/preds/anger-pred.txt ../../EmoInt/codalab/test_data/anger-gold.txt ../../kode/testkode/preds/fear-pred.txt ../../EmoInt/codalab/test_data/fear-gold.txt ../../kode/testkode/preds/joy-pred.txt ../../EmoInt/codalab/test_data/joy-gold.txt ../../kode/testkode/preds/sadness-pred.txt ../../EmoInt/codalab/test_data/sadness-gold.txt ;;
    "class") python3 eval.py preds.txt ;;
esac
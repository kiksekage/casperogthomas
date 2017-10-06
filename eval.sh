case $1 in
    "reg") python ../EmoInt/evaluate.py 4 ../kode/testkode/preds/anger_pred.txt ../EmoInt/codalab/test_data/anger-gold.txt ../kode/testkode/preds/fear_pred.txt ../EmoInt/codalab/test_data/fear-gold.txt ../kode/testkode/preds/joy_pred.txt ../EmoInt/codalab/test_data/joy-gold.txt ../kode/testkode/preds/sadness_pred.txt ../EmoInt/codalab/test_data/sadness-gold.txt ;;
    "class") python3 testkode/eval.py testkode/preds.txt ;;
esac
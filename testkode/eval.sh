case $1 in
    "reg") python ../../EmoInt/evaluate.py 4 ../../kode/testkode/preds/anger-pred.txt ../../kode/testkode/data18/reg/2018-EI-reg-En-dev/2018-EI-reg-En-anger-dev.txt ../../kode/testkode/preds/fear-pred.txt ../../kode/testkode/data18/reg/2018-EI-reg-En-dev/2018-EI-reg-En-fear-dev.txt ../../kode/testkode/preds/joy-pred.txt ../../kode/testkode/data18/reg/2018-EI-reg-En-dev/2018-EI-reg-En-joy-dev.txt ../../kode/testkode/preds/sadness-pred.txt ../../kode/testkode/data18/reg/2018-EI-reg-En-dev/2018-EI-reg-En-sadness-dev.txt ;;
    "class") python3 eval.py preds.txt ;;
esac
def compute_precision_recall(human_labels, model_labels):

    TP = FP = TN = FN = 0
    for i in xrange(len(human_labels)):
        if human_labels[i] == 1 and model_labels[i] == 1:
            TP +=1
        elif human_labels[i] == 1 and model_labels[i] == 0:
            FN += 1
        elif human_labels[i] == 0 and model_labels[i] == 1:
            FP += 1
        elif human_labels[i] == 0 and model_labels[i] == 0:
            TN += 1
        else:
            print 'ERROR: human_labels', human_labels[i], 'model_labels', model_labels[i]
    pre_1 = 1.0*TP/ (TP + FP) if TP+FP > 0 else 0
    rec_1 = 1.0*TP/ (TP + FN) if TP+FN > 0 else 0
    f1_1 = 2.0 * (pre_1*rec_1) / (pre_1+rec_1) if pre_1+rec_1 > 0 else 0

    TP = FP = TN = FN = 0
    for i in xrange(len(human_labels)):
        if human_labels[i] == 1 and model_labels[i] == 1:
            TN +=1
        elif human_labels[i] == 1 and model_labels[i] == 0:
            FP += 1
        elif human_labels[i] == 0 and model_labels[i] == 1:
            FN += 1
        elif human_labels[i] == 0 and model_labels[i] == 0:
            TP += 1
        else:
            print 'ERROR: human_labels', human_labels[i], 'model_labels', model_labels[i]
    pre_0 = 1.0*TP/ (TP + FP) if TP+FP > 0 else 0
    rec_0 = 1.0*TP/ (TP + FN) if TP+FN > 0 else 0
    f1_0 = 2.0 * (pre_0*rec_0) / (pre_0+rec_0) if pre_0+rec_0 > 0 else 0

    return (pre_1, rec_1, f1_1),(pre_0, rec_0, f1_0)




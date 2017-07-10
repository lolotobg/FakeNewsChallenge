#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels, labels=LABELS, relatedLabels = RELATED):
    score = 0.0

    cm_side = len(labels)
    cm = [0] * cm_side
    for i in range(0, cm_side):
        cm[i] = [0] * cm_side

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance in relatedLabels:
                score += 0.50
        if g_stance in relatedLabels and t_stance in relatedLabels:
            score += 0.25

        cm[labels.index(g_stance)][labels.index(t_stance)] += 1

    return score, cm


def print_confusion_matrix(cm, labels=LABELS,relatedLabels = RELATED):
    lines = []
    header = ("|{:^11}" + "|{:^11}"*len(labels) + "|").format('', *labels)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append(("|{:^11}" + "|{:^11}"*len(labels) + "|").format(labels[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))


def report_score(actual,predicted,labels=LABELS,relatedLabels = RELATED):
    score,cm = score_submission(actual,predicted,labels,relatedLabels)
    best_score, _ = score_submission(actual,actual,labels,relatedLabels)

    print_confusion_matrix(cm,labels,relatedLabels)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score


if __name__ == "__main__":
    actual = [0,0,0,0,1,1,0,3,3]
    predicted = [0,0,0,0,1,1,2,3,3]

    report_score([LABELS[e] for e in actual],[LABELS[e] for e in predicted])
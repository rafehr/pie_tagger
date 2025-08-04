import statistics

baseline = {
    "f1_scores": [
        0.8703703703703703,
        0.819277108433735,
        0.8860759493670886,
        0.8089887640449439,
        0.7723577235772359,
        0.9397590361445783,
        0.8240740740740741,
        0.8115942028985507,
        0.8949771689497716,
        0.9635627530364372
    ],
    "precision_scores": [
        0.8545454545454545,
        0.796875,
        0.9130434782608695,
        0.8059701492537313,
        0.7851239669421488,
        0.936,
        0.8640776699029126,
        0.8235294117647058,
        0.8990825688073395,
        0.952
    ],
    "recall_scores": [
        0.8867924528301887,
        0.8429752066115702,
        0.860655737704918,
        0.8120300751879699,
        0.76,
        0.9435483870967742,
        0.7876106194690266,
        0.8,
        0.8909090909090909,
        0.9754098360655737
    ],
    "class_reports": [],
    "mean_f1_score": 0.8591037150896785,
    "mean_precision_score": 0.8630247699477163,
    "mean_recall_score": 0.8559931405875112,
    "std_f1_score": 0.06206186744420102,
    "std_precision_score": 0.0599781992111869,
    "std_recall_score": 0.06931733079357343
}

sense = {
    "f1_scores": [
        0.8715596330275229,
        0.9456066945606695,
        0.7815126050420167,
        0.7164179104477612,
        0.7457627118644067,
        0.8925619834710744,
        0.7452830188679246,
        0.8640776699029127,
        0.8378378378378378,
        0.9672131147540983
    ],
    "precision_scores": [
        0.8482142857142857,
        0.9576271186440678,
        0.8017241379310345,
        0.7111111111111111,
        0.7927927927927928,
        0.9152542372881356,
        0.797979797979798,
        0.8811881188118812,
        0.8303571428571429,
        0.9672131147540983
    ],
    "recall_scores": [
        0.8962264150943396,
        0.9338842975206612,
        0.7622950819672131,
        0.7218045112781954,
        0.704,
        0.8709677419354839,
        0.6991150442477876,
        0.8476190476190476,
        0.8454545454545455,
        0.9672131147540983
    ],
    "class_reports": [],
    "mean_f1_score": 0.8367833179776225,
    "mean_precision_score": 0.8503461857884348,
    "mean_recall_score": 0.8248579799871372,
    "std_f1_score": 0.08699996357760864,
    "std_precision_score": 0.08065055323449336,
    "std_recall_score": 0.09733427027593564
}

syntax_0_1 = {
  "f1_scores": [
        0.0,
        0.9218106995884774,
        0.0,
        0.7876447876447876,
        0.8286852589641432,
        0.9098360655737704,
        0.7873303167420814,
        0.9238095238095239,
        0.8090909090909091,
        0.0
    ],
    "precision_scores": [
        0.0,
        0.9180327868852459,
        0.0,
        0.8095238095238095,
        0.8253968253968254,
        0.925,
        0.8055555555555556,
        0.9238095238095239,
        0.8090909090909091,
        0.0
    ],
    "recall_scores": [
        0.0,
        0.9256198347107438,
        0.0,
        0.7669172932330827,
        0.832,
        0.8951612903225806,
        0.7699115044247787,
        0.9238095238095239,
        0.8090909090909091,
        0.0
    ],
    "class_reports": [],
    "mean_f1_score": 0.5968207561413692,
    "mean_precision_score": 0.6016409410261868,
    "mean_recall_score": 0.5922510355591617,
    "std_f1_score": 0.4150811454171594,
    "std_precision_score": 0.41796751398348136,
    "std_recall_score": 0.412536182608838
}

syntax_0_5 = {
    "f1_scores": [
        0.5504587155963302,
        0.7795275590551181,
        0.8319327731092437,
        0.8060836501901141,
        0.7053941908713692,
        0.7773279352226721,
        0.8303571428571428,
        0.0,
        0.7129629629629629,
        0.8525896414342631
    ],
    "precision_scores": [
        0.5357142857142857,
        0.7443609022556391,
        0.853448275862069,
        0.8153846153846154,
        0.7327586206896551,
        0.7804878048780488,
        0.8378378378378378,
        0.0,
        0.7264150943396226,
        0.8294573643410853
    ],
    "recall_scores": [
        0.5660377358490566,
        0.8181818181818182,
        0.8114754098360656,
        0.7969924812030075,
        0.68,
        0.7741935483870968,
        0.8230088495575221,
        0.0,
        0.7,
        0.8770491803278688
    ],
    "class_reports": [],
    "mean_f1_score": 0.6846634571299216,
    "mean_precision_score": 0.6855864801302858,
    "mean_recall_score": 0.6846939023342435,
    "std_f1_score": 0.25635228777188473,
    "std_precision_score": 0.2577138427584513,
    "std_recall_score": 0.2568585093570667
}

syntax_1 = {
    "f1_scores": [
        0.867579908675799,
        0.0,
        0.8,
        0.0,
        0.728,
        0.6613545816733067,
        0.7610619469026548,
        0.0,
        0.0,
        0.8412698412698412
    ],
    "precision_scores": [
        0.8407079646017699,
        0.0,
        0.8135593220338984,
        0.0,
        0.728,
        0.6535433070866141,
        0.7610619469026548,
        0.0,
        0.0,
        0.8153846153846154
    ],
    "recall_scores": [
        0.8962264150943396,
        0.0,
        0.7868852459016393,
        0.0,
        0.728,
        0.6693548387096774,
        0.7610619469026548,
        0.0,
        0.0,
        0.8688524590163934
    ],
    "class_reports": [],
    "mean_f1_score": 0.4659266278521602,
    "mean_precision_score": 0.46122571560095527,
    "mean_recall_score": 0.4710380905624705,
    "std_f1_score": 0.4049887224869864,
    "std_precision_score": 0.400353629400012,
    "std_recall_score": 0.4103902116535358
}

baseline_dep_embs = {
    "f1_scores": [
        0.8857142857142858,
        0.9294605809128631,
        0.9106382978723404,
        0.751937984496124,
        0.8095238095238094,
        0.8605577689243028,
        0.8303571428571428,
        0.9607843137254903,
        0.8272727272727273,
        0.9140624999999999
    ],
    "precision_scores": [
        0.8942307692307693,
        0.9333333333333333,
        0.9469026548672567,
        0.776,
        0.8031496062992126,
        0.8503937007874016,
        0.8378378378378378,
        0.98989898989899,
        0.8272727272727273,
        0.8731343283582089
    ],
    "recall_scores": [
        0.8773584905660378,
        0.9256198347107438,
        0.8770491803278688,
        0.7293233082706767,
        0.816,
        0.8709677419354839,
        0.8230088495575221,
        0.9333333333333333,
        0.8272727272727273,
        0.9590163934426229
    ],
    "class_reports": [],
    "mean_f1_score": 0.8680309411299085,
    "mean_precision_score": 0.8732153947885738,
    "mean_recall_score": 0.8638949859417016
}

sense_enhanced_dep_embs = {    
    "f1_scores": [
        0.9295774647887324,
        0.902127659574468,
        0.8464730290456433,
        0.823529411764706,
        0.83739837398374,
        0.9011857707509883,
        0.831858407079646,
        0.8611111111111112,
        0.9,
        0.9547325102880658
    ],
    "precision_scores": [
        0.9252336448598131,
        0.9298245614035088,
        0.8571428571428571,
        0.860655737704918,
        0.8512396694214877,
        0.8837209302325582,
        0.831858407079646,
        0.8378378378378378,
        0.9,
        0.9586776859504132
    ],
    "recall_scores": [
        0.9339622641509434,
        0.8760330578512396,
        0.8360655737704918,
        0.7894736842105263,
        0.824,
        0.9193548387096774,
        0.831858407079646,
        0.8857142857142857,
        0.9,
        0.9508196721311475
    ],
    "class_reports": [],
    "mean_f1_score": 0.8787993738387101,
    "mean_precision_score": 0.8836191331633041,
    "mean_recall_score": 0.874728178361796
}


print('"std_f1_score":', statistics.stdev(baseline['f1_scores']))
print('"std_precision_score":', statistics.stdev(baseline['precision_scores']))
print('"std_recall_score":', statistics.stdev(baseline['recall_scores']))

print()

print('"std_f1_score":', statistics.stdev(sense['f1_scores']))
print('"std_precision_score":', statistics.stdev(sense['precision_scores']))
print('"std_recall_score":', statistics.stdev(sense['recall_scores']))

print()

print('"std_f1_score":', statistics.stdev(syntax_0_1['f1_scores']))
print('"std_precision_score":', statistics.stdev(syntax_0_1['precision_scores']))
print('"std_recall_score":', statistics.stdev(syntax_0_1['recall_scores']))

print()

print('"std_f1_score":', statistics.stdev(syntax_0_5['f1_scores']))
print('"std_precision_score":', statistics.stdev(syntax_0_5['precision_scores']))
print('"std_recall_score":', statistics.stdev(syntax_0_5['recall_scores']))

print()

print('"std_f1_score":', statistics.stdev(syntax_1['f1_scores']))
print('"std_precision_score":', statistics.stdev(syntax_1['precision_scores']))
print('"std_recall_score":', statistics.stdev(syntax_1['recall_scores']))

print()

print('"std_f1_score":', statistics.stdev(baseline_dep_embs['f1_scores']))
print('"std_precision_score":', statistics.stdev(baseline_dep_embs['precision_scores']))
print('"std_recall_score":', statistics.stdev(baseline_dep_embs['recall_scores']))

print()


print('"std_f1_score":', statistics.stdev(sense_enhanced_dep_embs['f1_scores']))
print('"std_precision_score":', statistics.stdev(sense_enhanced_dep_embs['precision_scores']))
print('"std_recall_score":', statistics.stdev(sense_enhanced_dep_embs['recall_scores']))

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'O', 'I-MISC', 'I-PER', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
print(f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

from openpyxl import load_workbook
from conceptnet_score.pathfinder import cal_qa_score, load_cpnet

load_cpnet()

wb = load_workbook('test_data_concept.xlsx')
ws = wb.active
rows = ws.rows
n = 0
scores = []
for row in rows:
    n += 1
    if n <= 5:
        line = [col.value for col in row]
        q_con = line[0].split('|')
        a1_con = line[1].split('|')
        a2_con = line[2].split('|')
        a3_con = line[3].split('|')
        a4_con = line[4].split('|')
        s1 = cal_qa_score(q_con, a1_con)
        s2 = cal_qa_score(q_con, a2_con)
        s3 = cal_qa_score(q_con, a3_con)
        s4 = cal_qa_score(q_con, a4_con)
        scores.append(str(s1)+','+str(s2)+','+str(s3)+','+str(s4))
        print(n, 'done..')

with open('concept_score.txt', 'w') as f:
    for i in scores:
        f.write(i+'\n')

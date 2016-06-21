import numpy as np
import DataUtility as du

class Student:
    hint_time_offset = 10
    hint_time_offset_std = 5
    ID = 0

    def __init__(self, hint, speed, knowledge, h_sd, s_sd, k_sd):
        self.hint = float(hint)
        self.speed = float(speed)
        self.knowledge = float(knowledge)
        self.hint_std = float(h_sd)
        self.speed_std = float(s_sd)
        self.knowledge_std = float(k_sd)
        self.ID = Student.ID
        Student.ID += 1

    def get_result(self, problem, pr_number):
        answer = du.clamp(np.random.normal(self.knowledge, self.knowledge_std), 0, 1)
        pr_hint = self.hint / pr_number

        hint = int(du.diceRoll(1000) < (pr_hint*1000))
        cor = int(problem < answer) * (1-hint)
        time = du.clamp(np.random.normal(self.speed, self.speed_std), 0, 10000) * (problem)
        time += du.MAX(0, np.random.normal(Student.hint_time_offset,
                                           Student.hint_time_offset_std)) * hint

        return [cor, time, hint]

    def do_assignment(self, skill):
        consecutive = 0
        problem_count = 0
        sequence = []
        complete = 1
        while consecutive < 3:
            result = self.get_result(skill.next_problem(),problem_count+1)
            sequence.append(result)
            if result[0] == 1:
                consecutive += 1
            else:
                consecutive = 0

            problem_count += 1

            if problem_count >= 50:
                complete = 0
                break

        means = np.mean(sequence,0)
        pcor = means[0]
        speed = means[1]
        hint = means[2]

        return [skill.name,self.ID,speed,pcor,problem_count,hint,complete]

class Skill:
    def __init__(self, name, difficulty=0.5, difficulty_std=0.1):
        self.problems = []
        self.name = name
        for i in range(0,10000):
            self.problems.append(du.clamp(np.random.normal(difficulty, difficulty_std), 0, 1))

    def next_problem(self):
        return self.problems[du.rand(0, len(self.problems))]

if __name__ == "__main__":
    data,headers = du.loadCSVwithHeaders('filtered_data.csv')

    students = []
    num_students = 100
    for i in range(0,num_students):
        index = du.rand(0,len(data))
        #self, hint, speed, knowledge, h_sd, s_sd, k_sd):
        students.append(Student(data[index][6],
                                data[index][1],
                                data[index][4],
                                data[index][7],
                                data[index][2],
                                du.clamp(np.random.normal(0.2, 0.1), 0.001, 1)))

    A = Skill('A',0.5,0.05)


    sim_header = ['Skill','Student','Speed','PercentCorrect','MasterySpeed',
                  'HintUsage','Complete']
    sim_data = []
    sim_data.append(sim_header)
    for i in range(0,num_students):
        sim_data.append(students[i].do_assignment(A))

    du.writetoCSV(sim_data,'simulated_data.csv')
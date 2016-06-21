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

        self.completed_assignments = []
        self.assignment_ID = 0

    def get_prereq_effects(self, skillName):
        knowledge_effect = 0
        speed_effect = 0
        hint_effect = 0

        for sk in SkillLink.list:
            if sk.postreq == skillName and du.exists(sk.prereq,self.completed_assignments):
                knowledge_effect += sk.knowledge_effect
                speed_effect += sk.speed_effect
                hint_effect += sk.hint_effect

        return [knowledge_effect, speed_effect, hint_effect]

    def get_result(self, problem, pr_number, skillName):
        effects = self.get_prereq_effects(skillName)

        knowledge_p = self.knowledge + effects[0]
        speed_p = self.speed + effects[1]
        hint_p = self.hint + effects[2]

        answer = du.clamp(np.random.normal(knowledge_p, self.knowledge_std), 0, 1)
        pr_hint = hint_p / pr_number

        hint = int(du.diceRoll(1000) < (pr_hint*1000))

        cor = int(problem < answer) * (1-hint)
        time = du.clamp(np.random.normal(speed_p, self.speed_std), 0, 10000) * problem
        time += du.MAX(0, np.random.normal(Student.hint_time_offset,
                                           Student.hint_time_offset_std)) * hint

        return [cor, time, hint]

    def do_assignment(self, skill):
        consecutive = 0
        problem_count = 0
        sequence = []
        complete = 1
        while consecutive < 3:
            result = self.get_result(skill.next_problem(),problem_count+1,skill.name)
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

        self.assignment_ID += 1
        self.completed_assignments.append(skill.name)

        return [skill.name, self.assignment_ID, self.ID, speed, pcor,
                problem_count, hint, complete]

    def clear_history(self):
        self.completed_assignments = []


class Skill:
    def __init__(self, name, difficulty=0.5, difficulty_std=0.1):
        self.problems = []
        self.name = name
        for i in range(0,10000):
            self.problems.append(du.clamp(np.random.normal(difficulty, difficulty_std), 0, 1))

    def next_problem(self):
        return self.problems[du.rand(0, len(self.problems))]


class SkillLink:
    list = []
    def __init__(self, prereq, postreq, knowledge_effect, speed_effect,hint_effect):
        self.prereq = prereq
        self.postreq = postreq
        self.knowledge_effect = knowledge_effect
        self.speed_effect = speed_effect
        self.hint_effect = hint_effect

        SkillLink.list.append(self)


if __name__ == "__main__":
    data, headers = du.loadCSVwithHeaders('filtered_data.csv')

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

    A = Skill('A', 0.5, 0.05)
    B = Skill('B', 0.6, 0.1)
    C = Skill('C', 0.3, 0.05)

    link = []

    # practice gives increase to knowledge and decrease to speed and hint usage
    link.append(SkillLink('A', 'A', 0.05, -2, -0.15))
    link.append(SkillLink('A', 'B', 0.1, -5, -0.05))

    sim_header = ['Skill', 'Assignment', 'Student', 'Speed', 'PercentCorrect',
                  'MasterySpeed', 'HintUsage', 'Complete']
    sim_data = []
    sim_data.append(sim_header)
    for i in range(0,num_students):
        students[i].clear_history()
        sim_data.append(students[i].do_assignment(A))
        sim_data.append(students[i].do_assignment(B))
        sim_data.append(students[i].do_assignment(A))
        sim_data.append(students[i].do_assignment(C))

    du.writetoCSV(sim_data, 'simulated_data.csv')
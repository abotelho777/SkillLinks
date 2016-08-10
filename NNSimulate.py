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
                knowledge_effect += sk.get_knowledge_effect()
                speed_effect += sk.get_speed_effect()
                hint_effect += sk.get_hint_effect()

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
    @staticmethod
    def get_hierarchy():
        sk_h = []
        for link in SkillLink.list:
            if link.prereq != link.postreq:
                sk_h.append(link)
        return sk_h

    @staticmethod
    def get_hierarchy_names():
        sk_h = []
        for link in SkillLink.list:
            if link.prereq != link.postreq:
                pre_post = [link.prereq,link.postreq]
                sk_h.append(pre_post)
        return sk_h

    def __init__(self, prereq, postreq, knowledge_effect, speed_effect,hint_effect,
                 knowledge_std, speed_std, hint_std):
        self.prereq = prereq
        self.postreq = postreq
        self.knowledge_effect = knowledge_effect
        self.speed_effect = speed_effect
        self.hint_effect = hint_effect
        self.knowledge_effect_std = knowledge_std
        self.speed_effect_std = speed_std
        self.hint_effect_std = hint_std

        SkillLink.list.append(self)

    def get_knowledge_effect(self):
        return du.MAX(np.random.normal(self.knowledge_effect, self.knowledge_effect_std), 0)

    def get_speed_effect(self):
        return du.MAX(np.random.normal(self.speed_effect, self.speed_effect_std), 0)

    def get_hint_effect(self):
        return du.MAX(np.random.normal(self.hint_effect, self.hint_effect_std), 0)


if __name__ == "__main__":
    data, headers = du.loadCSVwithHeaders('filtered_data.csv')

    students = []
    num_students = 100
    for i in range(0,num_students):
        index = du.rand(0,len(data))
        #self, hint, speed, knowledge, h_sd, s_sd, k_sd):
        students.append(Student(data[index][7],
                                data[index][1],
                                data[index][4],
                                data[index][8],
                                data[index][2],
                                data[index][5]))

    A = Skill('A', 0.5, 0.05)
    B = Skill('B', 0.6, 0.1)
    C = Skill('C', 0.3, 0.05)
    D = Skill('D', 0.6, 0.1)
    E = Skill('E', 0.8, 0.1)
    F = Skill('F', 0.4, 0.05)
    G = Skill('G', 0.6, 0.05)
    H = Skill('H', 0.7, 0.1)
    I = Skill('I', 0.7, 0.1)
    J = Skill('J', 0.5, 0.1)

    link = []

    # the effects of practice
    # prereq, postreq, knowledge_effect, speed_effect,hint_effect, std kn, std sp, std h
    link.append(SkillLink('A', 'A', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('B', 'B', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('C', 'C', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('D', 'D', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('E', 'E', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('F', 'F', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('G', 'G', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('H', 'H', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('I', 'I', 0.02, -1, -0.15, 0.01, 0.1, 0.02))
    link.append(SkillLink('J', 'J', 0.02, -1, -0.15, 0.01, 0.1, 0.02))

    # prerequisite relationships
    link.append(SkillLink('A', 'B', 0.1, -3, -0.05, 0.01, 0.1, 0.02))
    link.append(SkillLink('B', 'C', 0.2, -2, -0.1, 0.01, 0.1, 0.02))
    link.append(SkillLink('B', 'D', 0.1, -2, -0.05, 0.01, 0.3, 0.02))
    link.append(SkillLink('D', 'E', 0.2, -3, -0.05, 0.01, 0.1, 0.02))
    link.append(SkillLink('F', 'G', 0.1, -3, -0.05, 0.01, 0.1, 0.02))
    link.append(SkillLink('G', 'E', 0.1, -2, -0.05, 0.01, 0.1, 0.02))
    link.append(SkillLink('E', 'I', 0.3, -3, -0.05, 0.01, 0.1, 0.02))
    link.append(SkillLink('G', 'H', 0.1, -5, -0.05, 0.01, 0.1, 0.02))

    nonlink = []
    nonlink.append(['A', 'G'])
    nonlink.append(['A', 'F'])
    nonlink.append(['A', 'H'])
    nonlink.append(['B', 'J'])
    nonlink.append(['B', 'F'])
    nonlink.append(['J', 'A'])
    nonlink.append(['H', 'D'])
    nonlink.append(['H', 'C'])

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
        sim_data.append(students[i].do_assignment(B))
        sim_data.append(students[i].do_assignment(D))
        sim_data.append(students[i].do_assignment(J))
        sim_data.append(students[i].do_assignment(B))
        sim_data.append(students[i].do_assignment(F))
        sim_data.append(students[i].do_assignment(G))
        sim_data.append(students[i].do_assignment(H))
        sim_data.append(students[i].do_assignment(E))
        sim_data.append(students[i].do_assignment(F))
        sim_data.append(students[i].do_assignment(A))
        sim_data.append(students[i].do_assignment(D))
        sim_data.append(students[i].do_assignment(B))
        sim_data.append(students[i].do_assignment(I))
        sim_data.append(students[i].do_assignment(G))
        sim_data.append(students[i].do_assignment(E))
        sim_data.append(students[i].do_assignment(H))



    du.writetoCSV(sim_data, 'simulated_data')

    du.writetoCSV(SkillLink.get_hierarchy_names(), 'simulated_hierarchy')
    du.writetoCSV(nonlink, 'simulated_hierarchy_nonlink')
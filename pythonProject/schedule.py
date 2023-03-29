
# Path: schedule.py

# get four classes as list from dictionary
def get_classes(classes):
    # choose one class from each day in classes
    # classes is a dictionary with keys 1-5
    schedules = []
    for a in classes[1]:
        for b in classes[2]:
            for c in classes[3]:
                for d in classes[4]:
                    for e in classes[5]:
                        schedule = [a, b, c, d, e]
                        if valid(schedule):
                            schedules.append(schedule)
    return schedules

def valid(schedule):
    cs, math=0, 0
    can_audit = False
    contain_225 = False
    for course in schedule:
        if "MATH" in course:
            math += 1
        if "COMP" in course:
            cs += 1
        if "Audit" in course:
            can_audit = True
        if "COMP225" in course:
            contain_225 = True
    if can_audit:
        return cs >= 2 and math >=1 and contain_225
    else:
        return 'COMP225-02' not in schedule

def print_schedules(results):
    # print a table of schedules with time and class id and class name
    for i in range(len(results)):


def main():
    classes = {1: ['MATH377 (Audit)', 'COMP225-01'], 2: ['COMP446', 'COMP225-02'], 3: ["COMP435"],
               4: ["COMP456", "MATH437"], 5: ["MATH432"]}
    schedules = get_classes(classes)
    # print all schedules
    results = []
    for schedule in schedules:
        if len(schedule) > 4 and 'MATH377 (Audit)' not in schedule:
            subschedules = schedule[2:]
            # choose 2 class from subschedules
            for i in range(len(subschedules)):
                for j in range(i+1, len(subschedules)):
                    results.append(schedule[:2] + [subschedules[i], subschedules[j]])
        else:
            results.append(schedule)



main()

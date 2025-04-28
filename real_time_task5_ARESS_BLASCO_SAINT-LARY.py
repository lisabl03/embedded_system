# -*- coding: utf-8 -*-

"""

@author: ARESSY Alyssa, BLASCO Lisa, SAINT-LARY Samantha

Scheduling problem: Progressive job scheduling with two variants:
1. τ5 is executed even if it misses its deadline.
2. τ5 is dropped if it cannot meet its deadline.
To change the number of permutations display, change the value of top_k.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==============================
# Task Definitions
# ==============================

WCET = np.array([2, 3, 2, 2, 2, 2, 3])  # Worst-Case Execution Times
Deadlines = np.array([10, 10, 20, 20, 40, 40, 80])  # Deadlines for each task
IDs = np.array([1, 2, 3, 4, 5, 6, 7])  # Task IDs

hyperperiod = np.lcm.reduce(Deadlines)  # Least Common Multiple of deadlines
number_jobs = (hyperperiod / Deadlines).astype(int)  # Number of jobs per task
job_count = {tid: number_jobs[i] for i, tid in enumerate(IDs)}
total_jobs = sum(number_jobs)


# ==============================
# Case 1: τ5 Executed Even if Late
# ==============================

def generate_top_k_permutations_case1(task_ids, number_jobs_dict, WCET, Deadlines, top_k=3):
    results = []
    memo = {}
    total_jobs_local = sum(number_jobs_dict[tid] for tid in task_ids)

    def backtrack(current_perm, job_done, time, total_waiting, missed_deadlines):
        if len(current_perm) == total_jobs_local:
            results.append((total_waiting, missed_deadlines, tuple(current_perm)))
            return

        state_key = (tuple(sorted(job_done.items())), time)
        if state_key in memo and memo[state_key] <= total_waiting:
            return
        memo[state_key] = total_waiting

        candidates = []
        for tid in task_ids:
            idx = tid - 1
            if job_done[tid] < number_jobs_dict[tid]:
                job_num = job_done[tid] + 1
                release = (job_num - 1) * Deadlines[idx]
                deadline = job_num * Deadlines[idx]
                candidates.append((release, deadline, WCET[idx], tid, job_num))

        # Prioritize τ5
        candidates.sort(key=lambda x: (x[3] != 5, x[0], x[1]))

        for release, deadline, wcet, tid, job_num in candidates:
            idx = tid - 1
            start_time = max(time, release)
            finish_time = start_time + wcet
            waiting = start_time - release

            if finish_time > deadline and tid != 5:
                continue

            miss = 1 if (tid == 5 and finish_time > deadline) else 0

            job_done[tid] += 1
            current_perm.append((tid, job_num))
            backtrack(current_perm, job_done, finish_time, total_waiting + waiting, missed_deadlines + miss)
            current_perm.pop()
            job_done[tid] -= 1

    backtrack([], defaultdict(int), 0, 0, 0)

    filtered = [r for r in results if r[1] == 1]
    return sorted(filtered, key=lambda x: (x[0], x[1]))[:top_k]


# Progressive scheduling for Case 1
def progressive_schedule_case1(all_ids, WCET, Deadlines, job_count, top_k):
    current_ids = list(all_ids[:4])
    current_jobs = {tid: job_count[tid] for tid in current_ids}
    current_best = generate_top_k_permutations_case1(current_ids, current_jobs, WCET, Deadlines, top_k)

    for new_tid in all_ids[4:]:
        print(f"Adding task τ{new_tid}...")
        current_ids.append(new_tid)
        current_jobs = {tid: job_count[tid] for tid in current_ids}
        current_best = generate_top_k_permutations_case1(current_ids, current_jobs, WCET, Deadlines, top_k)

    return current_best[::-1]


# Visualization
def visualize_schedule(permutation, WCET, Deadlines, task_ids, case_title):
    task_colors = plt.cm.get_cmap('tab10', len(task_ids))
    timeline = []
    time = 0

    for task_id, job_num in permutation:
        idx = task_id - 1
        release = (job_num - 1) * Deadlines[idx]
        start_time = max(time, release)
        end_time = start_time + WCET[idx]
        missed = (task_id == 5 and end_time > job_num * Deadlines[idx])
        timeline.append((task_id, job_num, start_time, end_time, missed))
        time = end_time

    fig, ax = plt.subplots(figsize=(12, 4))
    task_positions = {tid: (len(task_ids) - i - 1) * 10 for i, tid in enumerate(task_ids)}

    for task, job, start, end, missed in timeline:
        color = 'red' if missed else task_colors(task - 1)
        y_pos = task_positions[task]
        ax.broken_barh([(start, end - start)], (y_pos, 9), facecolors=color)
        ax.text(start + (end - start)/2, y_pos + 4.5,
                f"T{task}J{job}", ha='center', va='center', color='white', fontsize=8)

    ax.set_yticks([task_positions[tid] + 4.5 for tid in task_ids])
    ax.set_yticklabels([f'Task {tid}' for tid in task_ids])
    ax.set_xlabel("Time")
    ax.set_title(case_title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Execute Case 1
final_results_case1 = progressive_schedule_case1(IDs, WCET, Deadlines, job_count, top_k=5)

print("\nFinal results (Case 1):")
if final_results_case1:
    for i, (w, m, p) in enumerate(final_results_case1):
        print(f"#{i+1} | Total waiting time: {w} | Missed deadlines (τ5): {m} | Sequence: {p}")
    best_waiting_time, missed, best_perm = final_results_case1[-1]
    visualize_schedule(best_perm, WCET, Deadlines, IDs, "Schedule with τ5 delayed (red)")
else:
    print("No sequence found for Case 1.")

#%%
# ==============================
# Case 2: τ5 Skipped if Late
# ==============================

def generate_top_k_permutations_case2(task_ids, number_jobs_dict, WCET, Deadlines, top_k=3):
    results = []
    memo = {}
    total_jobs_local = sum(number_jobs_dict[tid] for tid in task_ids)

    def backtrack(current_perm, job_done, time, total_waiting):
        if sum(job_done[tid] for tid in task_ids) == total_jobs_local:
            results.append((total_waiting, tuple(current_perm)))
            return

        state_key = (tuple(sorted(job_done.items())), time)
        if state_key in memo and memo[state_key] <= total_waiting:
            return
        memo[state_key] = total_waiting

        candidates = []
        for tid in task_ids:
            idx = tid - 1
            if job_done[tid] < number_jobs_dict[tid]:
                job_num = job_done[tid] + 1
                release = (job_num - 1) * Deadlines[idx]
                deadline = job_num * Deadlines[idx]
                candidates.append((release, deadline, WCET[idx], tid, job_num))

        candidates.sort(key=lambda x: (x[0], x[1]))

        for release, deadline, wcet, tid, job_num in candidates:
            idx = tid - 1
            start_time = max(time, release)
            finish_time = start_time + wcet
            waiting = start_time - release

            if finish_time > deadline:
                if tid == 5:
                    job_done[tid] += 1  # Skip τ5 job
                    backtrack(current_perm, job_done, time, total_waiting)
                    job_done[tid] -= 1
                continue

            job_done[tid] += 1
            current_perm.append((tid, job_num))
            backtrack(current_perm, job_done, finish_time, total_waiting + waiting)
            current_perm.pop()
            job_done[tid] -= 1

    backtrack([], defaultdict(int), 0, 0)
    return sorted(results, key=lambda x: x[0])[:top_k]


# Progressive scheduling for Case 2
def progressive_schedule_case2(all_ids, WCET, Deadlines, job_count, top_k):
    current_ids = list(all_ids[:4])
    current_jobs = {tid: job_count[tid] for tid in current_ids}
    current_best = generate_top_k_permutations_case2(current_ids, current_jobs, WCET, Deadlines, top_k)

    for new_tid in all_ids[4:]:
        print(f"Adding task τ{new_tid}...")
        current_ids.append(new_tid)
        current_jobs = {tid: job_count[tid] for tid in current_ids}
        current_best = generate_top_k_permutations_case2(current_ids, current_jobs, WCET, Deadlines, top_k)

    return current_best[::-1]


# Execute Case 2
final_results_case2 = progressive_schedule_case2(IDs, WCET, Deadlines, job_count, top_k=5)

print("\nFinal results (Case 2):")
if final_results_case2:
    for i, (w, p) in enumerate(final_results_case2):
        print(f"#{i+1} | Total waiting time: {w} | Sequence: {p}")
    best_waiting_time, best_perm = final_results_case2[-1]
    visualize_schedule(best_perm, WCET, Deadlines, IDs, "Schedule (τ5 skipped if late)")
else:
    print("No sequence found for Case 2.")

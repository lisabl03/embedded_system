# -*- coding: utf-8 -*-
#@author: ARESSY Alyssa, BLASCO Lisa, SAINT-LARY Samantha


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# To change the number of permutations display, change the value of top_k.

# =======================
# Task Definition
# =======================

# WCET (Worst-Case Execution Time): maximum execution time for each task
WCET = np.array([2, 3, 2, 2, 2, 2, 3])

# Relative deadlines associated with each task (in time units)
Deadlines = np.array([10, 10, 20, 20, 40, 40, 80])

# Task identifiers (τ1 to τ7)
IDs = np.array([1, 2, 3, 4, 5, 6, 7])

# Calculate the hyperperiod (smallest common multiple of the deadlines)
hyperperiod = np.lcm.reduce(Deadlines)

# Number of instances (jobs) each task must execute within the hyperperiod
number_jobs = (hyperperiod / Deadlines).astype(int)

# Dictionary associating each task with its number of jobs
job_count = {task_id: number_jobs[i] for i, task_id in enumerate(IDs)}

# Total number of jobs to be scheduled
total_jobs = sum(number_jobs)


# =======================
# Backtracking (full exploration)
# =======================
def generate_top_k_permutations(task_ids, number_jobs_dict, WCET, Deadlines, top_k=3):
    """
    Recursively explore all possible sequences for a subset of tasks,
    respecting their deadlines, and return the `top_k` sequences
    that minimize the total waiting time.
    """
    results = []  # List of valid permutations found
    memo = {}     # Memoization to avoid recalculations
    total_jobs_local = sum(number_jobs_dict[tid] for tid in task_ids)

    def backtrack(current_perm, job_done, time, total_waiting):
        # Terminal case: all jobs have been placed
        if len(current_perm) == total_jobs_local:
            results.append((total_waiting, tuple(current_perm)))
            return

        # Memorization to avoid suboptimal branches already seen
        state_key = (tuple(sorted(job_done.items())), time)
        if state_key in memo and memo[state_key] <= total_waiting:
            return
        memo[state_key] = total_waiting

        # Generate available candidate jobs at this moment
        candidates = []
        for task_id in task_ids:
            idx = task_id - 1
            if job_done[task_id] < number_jobs_dict[task_id]:
                job_num = job_done[task_id] + 1
                release = (job_num - 1) * Deadlines[idx]  # release time
                deadline = job_num * Deadlines[idx]       # absolute deadline
                candidates.append((release, deadline, WCET[idx], task_id, job_num))

        # Sort candidates: earliest release, tightest deadline first
        candidates.sort(key=lambda x: (x[0], x[1]))

        # Explore each candidate
        for release, deadline, wcet, task_id, job_num in candidates:
            idx = task_id - 1
            start_time = max(time, release)  # task can start at its release or later
            finish_time = start_time + WCET[idx]
            waiting = start_time - release

            # Check if deadline is respected
            if finish_time > deadline:
                continue

            # Recursive backtracking
            job_done[task_id] += 1
            current_perm.append((task_id, job_num))
            backtrack(current_perm, job_done, finish_time, total_waiting + waiting)
            current_perm.pop()
            job_done[task_id] -= 1

    # Start exploration
    backtrack([], defaultdict(int), 0, 0)

    # Return the `top_k` permutations with the smallest waiting time
    return sorted(results, key=lambda x: x[0])[:top_k]


# =======================
# Progressive Scheduling
# =======================
def progressive_schedule(all_ids, WCET, Deadlines, job_count, top_k=10):
    """
    Schedule tasks progressively:
    - Start with the first 4 tasks,
    - Add the next ones one by one,
    - At each step, recalculate the globally best `top_k` sequences possible.
    """
    # Starting tasks
    current_ids = list(all_ids[:4])
    current_jobs = {tid: job_count[tid] for tid in current_ids}

    # Initial generation of best permutations
    current_best = generate_top_k_permutations(current_ids, current_jobs, WCET, Deadlines, top_k=top_k)

    # Progressive addition of remaining tasks
    for new_tid in all_ids[4:]:
        print(f"Adding task τ{new_tid}...")
        current_ids.append(new_tid)
        current_jobs = {tid: job_count[tid] for tid in current_ids}
        current_best = generate_top_k_permutations(current_ids, current_jobs, WCET, Deadlines, top_k=top_k)

    # Reverse the order to have the best one last (optional)
    return current_best[::-1]


# =======================
# Script Usage
# =======================
final_results = progressive_schedule(IDs, WCET, Deadlines, job_count, top_k=1)

# Display results
print("\nFinal Results:")
for i, (w, p) in enumerate(final_results):
    print(f"#{i+1} | Total waiting time: {w} | Sequence: {p}")


def visualize_schedule(permutation, WCET, Deadlines, task_ids):
    task_colors = plt.colormaps.get_cmap('tab10')
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
    task_positions = {task_id: (len(task_ids) - i - 1) * 10 for i, task_id in enumerate(task_ids)}

    for task, job, start, end, missed in timeline:
        color = 'red' if missed else task_colors((task - 1) % 10)
        y_pos = task_positions[task]
        ax.broken_barh([(start, end - start)], (y_pos, 9), facecolors=color)
        ax.text(start + (end - start)/2, y_pos + 4.5,
                f"T{task}J{job}", ha='center', va='center', color='white', fontsize=8)

    ax.set_yticks([task_positions[tid] + 4.5 for tid in task_ids])
    ax.set_yticklabels([f'Task {tid}' for tid in task_ids])
    ax.set_xlabel("Time")
    ax.set_title("Schedule")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Visualize the best permutation
best_waiting_time, best_perm = final_results[-1]
visualize_schedule(best_perm, WCET, Deadlines, IDs)

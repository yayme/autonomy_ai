import os
import sys
import time
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid as cumtrapz
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# --------------------
# System Prompts
# --------------------
PROMPT_FULL = """
You are a helpful tutor. The student has submitted a problem.
Provide a complete, clearly structured, step-by-step solution.
Show all calculations and reasoning explicitly.
"""

PROMPT_PARTIAL = """
You are a Socratic tutor. The student has submitted a problem.
Do NOT give the full solution. Instead:
1. Identify the key concept needed to solve the problem and state it briefly.
2. Solve only the first part or first step as a worked example.
3. For the remaining parts, provide only the problem setup and the first step,
    then stop and ask the student one specific guiding question to help them
    proceed on their own.
Your goal is to keep the student actively thinking at every stage.
"""

PROMPT_DEFER = """
You are a foundational tutor. The student has submitted a problem,
but they need to strengthen their prerequisite understanding before
attempting it.

Do NOT engage with the submitted problem directly.

Instead, do the following:
1. Read the problem and identify the single most important prerequisite
    concept the student needs in order to approach it.
2. Teach that prerequisite concept in 3 to 4 sentences using one concrete
    numerical example that is entirely unrelated to the submitted problem.
3. Give the student one simple exercise on that prerequisite concept and
    ask them to solve it before returning to their original problem.

Do not reveal any part of the original problem's solution under any
circumstances.
"""

USER_QUERY = """
A frog sits on one of three lily pads labeled 1, 2, and 3.
Each minute, it jumps to one of the other two pads with equal probability.
The transition matrix is:

    P = [[0,   0.5, 0.5],
         [0.5, 0,   0.5],
         [0.5, 0.5, 0  ]]

(a) Starting from pad 1, what is the probability of being on pad 2 after 2 steps?
(b) What is the steady-state distribution of this Markov chain?
(c) Starting from pad 1, what is the expected number of steps to reach pad 3?
"""

LEVELS = [
    (0.0, "Full Assistance", PROMPT_FULL),
    (0.5, "Partial Assistance", PROMPT_PARTIAL),
    (1.0, "Full Deferral", PROMPT_DEFER),
]

# --------------------
# LLM Query Function (Google Gemini)
# --------------------
def query_llm(system_prompt: str, user_query: str) -> str:
    try:
        m = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            system_instruction=system_prompt
        )
        response = m.generate_content(user_query)
        return response.text
    except Exception as e:
        return f"[ERROR] LLM API call failed: {e}"

# --------------------
# Main LLM Query Loop
# --------------------
def run_llm_queries():
    responses = []
    for u, name, prompt in LEVELS:
        print(f"\n{'='*40}\nAssistance Level: {name} (u = {u})\n{'='*40}")
        resp = query_llm(prompt, USER_QUERY)
        print(resp + "\n")
        responses.append((u, name, resp))
    # Save to file
    out_path = os.path.join(os.path.dirname(__file__), "apai_responses.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for u, name, resp in responses:
            f.write(f"\n{'='*60}\nAssistance Level: {name} (u = {u})\n{'='*60}\n")
            f.write(resp + "\n\n")
    print(f"Responses saved to {out_path}")
    return responses

# --------------------
# ODE Right-Hand Side
# --------------------
def ode_rhs(t, c, u_func: Callable[[np.ndarray], np.ndarray], params: dict):
    mu = params['mu']
    alpha = params['alpha']
    lam = params['lam']
    gamma_c = params['gamma_c']
    # c: [c1, c2, c3]
    u = u_func(c)
    Lambda = np.zeros(3)
    Lambda[0] = lam
    Lambda[1] = lam + gamma_c * (1 - c[0])
    Lambda[2] = lam + gamma_c * (1 - c[1])
    dc = np.zeros(3)
    for v in range(3):
        dc[v] = u[v] * (mu * c[v]**alpha + Lambda[v]) - Lambda[v]
    return dc

# --------------------
# Policy Functions
# --------------------
def policy_full_ai(c):
    return np.zeros(3)

def policy_full_human(c):
    return np.ones(3)

def policy_apai(c, params):
    mu = params['mu']
    alpha = params['alpha']
    lam = params['lam']
    gamma_c = params['gamma_c']
    gamma_u = params['gamma_u']
    beta = params['beta']
    Lambda = np.zeros(3)
    Lambda[0] = lam
    Lambda[1] = lam + gamma_c * (1 - c[0])
    Lambda[2] = lam + gamma_c * (1 - c[1])
    u = np.zeros(3)
    for v in range(3):
        denom = beta * (mu * c[v]**alpha + Lambda[v])
        if denom == 0:
            u[v] = 1.0
        else:
            val = 1 - (gamma_u / denom)**(1/(gamma_u - 1))
            u[v] = np.clip(val, 0, 1)
    return u

# --------------------
# Simulation and Plotting
# --------------------
def simulate_and_plot():
    # Parameters
    params = dict(
        mu=0.3,
        lam=0.1,
        alpha=0.6,
        gamma_c=0.05,
        gamma_u=2.0,
        beta=1.5,
    )
    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 1000)
    c0 = np.array([0.4, 0.4, 0.4])

    # Policy A: Full AI
    sol_ai = solve_ivp(
        lambda t, c: ode_rhs(t, c, policy_full_ai, params),
        t_span, c0, t_eval=t_eval, vectorized=False
    )
    # Policy B: Full Human
    sol_human = solve_ivp(
        lambda t, c: ode_rhs(t, c, policy_full_human, params),
        t_span, c0, t_eval=t_eval, vectorized=False
    )
    # Policy C: APAI
    def apai_func(c):
        return policy_apai(c, params)
    sol_apai = solve_ivp(
        lambda t, c: ode_rhs(t, c, apai_func, params),
        t_span, c0, t_eval=t_eval, vectorized=False
    )

    # Compute c_star (atrophy threshold)
    u_bar = 0.5
    c_star = (params['lam'] * (1 - u_bar) / (params['mu'] * u_bar))**(1/params['alpha'])

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Capability Trajectories
    ax = axes[0]
    colors_ai = ['#d62728', '#ff9896', '#c93030']  # reds
    colors_human = ['#2ca02c', '#98df8a', '#1f7a1f']  # greens
    colors_apai = ['#1f77b4', '#aec7e8', '#174a7a']  # blues
    labels = ['Arithmetic (v1)', 'Probability (v2)', 'Markov Chain (v3)']
    # Full AI
    for i in range(3):
        ax.plot(sol_ai.t, sol_ai.y[i], color=colors_ai[i], label=f'Full AI: {labels[i]}', linestyle='-')
    # Full Human
    for i in range(3):
        ax.plot(sol_human.t, sol_human.y[i], color=colors_human[i], label=f'Full Human: {labels[i]}', linestyle='-')
    # APAI
    for i in range(3):
        ax.plot(sol_apai.t, sol_apai.y[i], color=colors_apai[i], label=f'APAI: {labels[i]}', linestyle='-')
    # Atrophy threshold
    ax.axhline(c_star, color='k', linestyle='--', label=f'atrophy threshold c*')
    ax.set_title('Capability Trajectories under Three Policies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Capability c_v(t)')
    ax.legend(fontsize=8, loc='lower right', ncol=2)

    # Subplot 2: Cumulative Task Utility
    ax2 = axes[1]
    gamma_u = params['gamma_u']
    # For each policy, compute CU(t) = \int_0^t sum_v (1-u_v(tau))^gamma_u dtau
    def compute_cu(sol, u_func):
        u_vals = np.array([u_func(sol.y[:,i]) for i in range(sol.y.shape[1])])  # shape (N, 3)
        util = np.sum((1 - u_vals)**gamma_u, axis=1)  # shape (N,)
        cu = np.concatenate([[0], cumtrapz(util, sol.t)])
        return cu
    cu_ai = compute_cu(sol_ai, policy_full_ai)
    cu_human = compute_cu(sol_human, policy_full_human)
    cu_apai = compute_cu(sol_apai, apai_func)
    # Plot
    ax2.plot(sol_ai.t, cu_ai, color='#d62728', label='Full AI Delegation')
    ax2.plot(sol_human.t, cu_human, color='#2ca02c', label='Full Human Engagement')
    ax2.plot(sol_apai.t, cu_apai, color='#1f77b4', label='APAI Policy')
    ax2.set_title('Cumulative Task Utility')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cumulative Utility')
    ax2.legend()

    plt.tight_layout()
    # Save
    outdir = os.path.dirname(__file__)
    pdf_path = os.path.join(outdir, 'apai_simulation.pdf')
    png_path = os.path.join(outdir, 'apai_simulation.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    print(f"Saved plots to {pdf_path} and {png_path}")

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    print("--- APAI Proof of Concept ---")
    responses = run_llm_queries()
    print("\n--- Running Capability Trajectory Simulation ---\n")
    simulate_and_plot()
    print("\nDone.")

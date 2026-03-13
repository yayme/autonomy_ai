import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

alpha = 0.6
gamma_c = 0.05
Q = 0.01
sigma = 0.15
T = 30
u_bar = 0.5
domains = ['Arithmetic', 'Probability', 'Markov Chain']
c0_true = np.array([0.6, 0.5, 0.4])
c0_hat = np.array([0.5, 0.5, 0.5])
P0 = np.array([0.1, 0.1, 0.1])

mu_vec  = np.array([0.30, 0.25, 0.15])
lam_vec = np.array([0.10, 0.10, 0.18])
c_star = (lam_vec[0] * (1 - u_bar) / (mu_vec[0] * u_bar)) ** (1 / alpha)
c_star_vec = np.array([
    (lam_vec[v] * (1 - u_bar) / (mu_vec[v] * u_bar)) ** (1 / alpha)
    for v in range(3)
])

COLORS = {'Full AI': '#d62728', 'APAI': '#1f77b4', 'Full Human': '#2ca02c'}

def phi(l_t, d_t, e_t):
    norm_l = 1 - np.clip(l_t / 30.0, 0, 1)
    norm_d = np.clip(d_t, 0, 1)
    return 0.5 * norm_l + 0.3 * norm_d + 0.2 * e_t

def schedule_full_ai(_c_hat):
    return np.full(3, 0.0)

def schedule_full_human(_c_hat):
    return np.full(3, 1.0)

def schedule_apai(c_hat):
    u = np.full(3, 0.5)
    for v in range(3):
        if c_hat[v] < c_star_vec[v] - 0.05:
            u[v] = 0.55 + 0.35 * max(0, (c_star_vec[v] - c_hat[v])) / c_star_vec[v]
    return u

def Lambda_vec(c):
    return np.array([
        lam_vec[0],
        lam_vec[1] + gamma_c * (1 - c[0]),
        lam_vec[2] + gamma_c * (1 - c[1]),
    ])

def capability_rhs(t, c, u):
    Lambda = Lambda_vec(c)
    dc = np.zeros(3)
    for v in range(3):
        c_safe = np.clip(c[v], 0, 1)
        dc[v] = u[v] * (mu_vec[v] * c_safe ** alpha + Lambda[v]) - Lambda[v]
    return np.array(dc)

def ekf_predict(c_hat, P, u):
    Lambda = Lambda_vec(c_hat)
    f = np.zeros(3)
    df_dc = np.zeros(3)
    for v in range(3):
        c_safe = np.clip(c_hat[v], 1e-6, 1)
        f[v] = u[v] * (mu_vec[v] * c_safe ** alpha + Lambda[v]) - Lambda[v]
        if c_safe > 0:
            df_dc[v] = u[v] * mu_vec[v] * alpha * c_safe ** (alpha - 1)
        else:
            df_dc[v] = 0.0
    sigma2 = sigma ** 2
    P_safe = np.clip(P, 1e-6, None)
    P_pred = P_safe + (2 * df_dc * P_safe - P_safe ** 2 / sigma2 + Q)
    P_pred = np.clip(P_pred, 1e-6, 5 * sigma ** 2)
    c_hat_pred = c_hat + f
    c_hat_pred = np.clip(c_hat_pred, 0.0, 1.0)
    return c_hat_pred, P_pred

def ekf_update(c_hat_pred, P_pred, o_t, v):
    sigma2 = sigma ** 2
    denom = P_pred[v] + sigma2
    K = P_pred[v] / denom if denom > 1e-6 else 0.0
    c_hat = c_hat_pred.copy()
    c_hat[v] = c_hat_pred[v] + K * (o_t - c_hat_pred[v])
    P = P_pred.copy()
    P[v] = max((1 - K) * P_pred[v], 1e-6)
    c_hat = np.clip(c_hat, 0.0, 1.0)
    return c_hat, P

schedules = [
    ('Full AI', schedule_full_ai, COLORS['Full AI']),
    ('APAI', schedule_apai, COLORS['APAI']),
    ('Full Human', schedule_full_human, COLORS['Full Human'])
]

results = []
for name, sched_func, color in schedules:
    c_true = c0_true.copy()
    c_hat = c0_hat.copy()
    P = P0.copy()
    c_true_hist = [c_true.copy()]
    c_hat_hist = [c_hat.copy()]
    P_hist = [P.copy()]
    u_hist = [sched_func(c_hat)]
    obs_hist  = [[] for _ in range(3)]
    obs_times = [[] for _ in range(3)]
    for step in range(T):
        v = step % 3
        u = sched_func(c_hat)
        sol = solve_ivp(lambda t, c, u=u: capability_rhs(t, c, u), [0, 1], c_true, method='RK45', t_eval=[1])
        c_true = np.clip(sol.y[:, -1], 0, 1)
        u = u.copy()
        for _v in range(3):
            if c_true[_v] < 1e-3:
                u[_v] = 0.0
        l_t = max(0, 30 * (1 - c_true[v]) + np.random.normal(0, 2))
        d_t = np.clip(c_true[v] + np.random.normal(0, 0.1), 0, 1)
        e_t = np.clip(c_true[v] * 0.8 + np.random.normal(0, 0.1), 0, 1)
        o_t = phi(l_t, d_t, e_t) + np.random.normal(0, sigma)
        o_t = np.clip(o_t, 0, 1)
        c_hat_pred, P_pred = ekf_predict(c_hat, P, u)
        P_pred = np.maximum(P_pred, 1e-6)
        c_hat, P = ekf_update(c_hat_pred, P_pred, o_t, v)
        c_true_hist.append(c_true.copy())
        c_hat_hist.append(c_hat.copy())
        P_hist.append(P.copy())
        u_hist.append(u.copy())
        obs_hist[v].append(o_t)
        obs_times[v].append(step + 1)
    results.append({
        'name': name,
        'color': color,
        'c_true_hist': np.array(c_true_hist),
        'c_hat_hist': np.array(c_hat_hist),
        'P_hist': np.array(P_hist),
        'u_hist': np.array(u_hist),
        'obs_hist': obs_hist,
        'obs_times': obs_times
    })

t = np.arange(T + 1)
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("APAI: Capability Estimation Proof of Concept", fontsize=15, fontweight='bold', y=1.01)

for v in range(3):
    ax = axes[0, v]
    for res in results:
        color = res['color']
        c_true = res['c_true_hist'][:, v]
        c_hat = res['c_hat_hist'][:, v]
        P = res['P_hist'][:, v]
        std = np.sqrt(np.clip(P, 0, None))
        half_band = np.minimum(1.96 * std, 0.15)
        ax.fill_between(t, np.clip(c_hat - half_band, 0, 1), np.clip(c_hat + half_band, 0, 1), alpha=0.10, color=color)
        ax.plot(t, c_true, color=color, linewidth=2.2, label=f"{res['name']} (true)")
        ax.plot(t, c_hat, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{res['name']} (EKF)")
        ax.scatter(res['obs_times'][v], res['obs_hist'][v], color=color, s=12, alpha=0.5)
    ax.axhline(c_star_vec[v], color='k', linestyle='--', linewidth=0.9, label='$c^*$')
    ax.set_title(domains[v], fontweight='bold', fontsize=14)
    ax.set_xlabel('Interaction step', fontsize=12)
    ax.set_ylabel('Capability $c_v(t)$', fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    if v == 2:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=10, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

ax = axes[1, 0]
for res in results:
    color = res['color']
    ax.plot(t, res['u_hist'][:, 2], color=color, linewidth=2, label=res['name'])
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='APAI base')
ax.set_title('Assistance Deterrent Level $u_{v_3}(t)$', fontsize=13)
ax.set_xlabel('Interaction step', fontsize=12)
ax.set_ylabel('$u_{v_3}(t)$', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)

ax = axes[1, 1]
for res in results:
    color = res['color']
    Lambda_v2 = [Lambda_vec(c)[1] for c in res['c_true_hist']]
    ax.plot(t, Lambda_v2, color=color, linewidth=2, label=res['name'])
ax.axhline(lam_vec[1], color='gray', linestyle=':', linewidth=1, label='baseline $\\lambda$')
ax.set_title('Cascade Effect $\\Lambda_{v_2}(t)$', fontsize=13)
ax.set_xlabel('Interaction step', fontsize=12)
ax.set_ylabel('$\\Lambda_{v_2}$', fontsize=12)
ax.legend(fontsize=10)

ax = axes[1, 2]
for res in results:
    color = res['color']
    ax.plot(t, res['P_hist'][:, 2], color=color, linewidth=2, label=res['name'])
obs_times_v3 = [step + 1 for step in range(T) if step % 3 == 2]
for obs_t in obs_times_v3:
    ax.axvline(obs_t, color='gray', alpha=0.2, linewidth=0.8)
ax.set_title('EKF Uncertainty $P_{v_3}(t)$', fontsize=13)
ax.set_xlabel('Interaction step', fontsize=12)
ax.set_ylabel('$P_{v_3}(t)$', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
fig.savefig('apai_estimation_poc.pdf', bbox_inches='tight')
fig.savefig('apai_estimation_poc.png', dpi=150, bbox_inches='tight')
plt.show()

# Save each subplot as a separate file
for v in range(3):
    fig_single, ax_single = plt.subplots(figsize=(6, 4))
    for res in results:
        color = res['color']
        c_true = res['c_true_hist'][:, v]
        c_hat = res['c_hat_hist'][:, v]
        P = res['P_hist'][:, v]
        std = np.sqrt(np.clip(P, 0, None))
        half_band = np.minimum(1.96 * std, 0.15)
        ax_single.fill_between(t, np.clip(c_hat - half_band, 0, 1), np.clip(c_hat + half_band, 0, 1), alpha=0.10, color=color)
        ax_single.plot(t, c_true, color=color, linewidth=2.2, label=f"{res['name']} (true)")
        ax_single.plot(t, c_hat, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{res['name']} (EKF)")
        ax_single.scatter(res['obs_times'][v], res['obs_hist'][v], color=color, s=20, alpha=0.5)
    ax_single.axhline(c_star_vec[v], color='k', linestyle='--', linewidth=0.9, label='$c^*$')
    ax_single.set_title(domains[v], fontweight='bold', fontsize=14)
    ax_single.set_xlabel('Interaction step', fontsize=12)
    ax_single.set_ylabel('Capability $c_v(t)$', fontsize=12)
    ax_single.set_ylim(-0.05, 1.1)
    if v == 2:
        handles, labels = ax_single.get_legend_handles_labels()
        ax_single.legend(handles, labels, fontsize=10, ncol=1, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    fig_single.tight_layout()
    fig_single.savefig(f'apai_capability_{domains[v].replace(" ", "_").lower()}.png', dpi=150)
    plt.close(fig_single)

# Row 2, (1,0): Assistance deterrent level u_v3 (Markov Chain)
fig_u, ax_u = plt.subplots(figsize=(6, 4))
for res in results:
    color = res['color']
    ax_u.plot(t, res['u_hist'][:, 2], color=color, linewidth=2, label=res['name'])
ax_u.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='APAI base')
ax_u.set_title('Assistance Deterrent Level $u_{v_3}(t)$', fontsize=13)
ax_u.set_xlabel('Interaction step', fontsize=12)
ax_u.set_ylabel('$u_{v_3}(t)$', fontsize=12)
ax_u.set_ylim(-0.05, 1.05)
ax_u.legend(fontsize=10)
fig_u.tight_layout()
fig_u.savefig('apai_assistance_deterrent_v3.png', dpi=150)
plt.close(fig_u)

# Also update the main grid plot
ax = axes[1, 0]
for res in results:
    color = res['color']
    ax.plot(t, res['u_hist'][:, 2], color=color, linewidth=2, label=res['name'])
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='APAI base')
ax.set_title('Assistance Deterrent Level $u_{v_3}(t)$', fontsize=13)
ax.set_xlabel('Interaction step', fontsize=12)
ax.set_ylabel('$u_{v_3}(t)$', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=10)

# Row 2, (1,1): Effective decay rate Lambda_v2
fig_lam, ax_lam = plt.subplots(figsize=(6, 4))
for res in results:
    color = res['color']
    Lambda_v2 = [Lambda_vec(c)[1] for c in res['c_true_hist']]
    ax_lam.plot(t, Lambda_v2, color=color, linewidth=2, label=res['name'])
ax_lam.axhline(lam_vec[1], color='gray', linestyle=':', linewidth=1, label='baseline $\\lambda$')
ax_lam.set_title('Cascade Effect $\\Lambda_{v_2}(t)$', fontsize=13)
ax_lam.set_xlabel('Interaction step', fontsize=12)
ax_lam.set_ylabel('$\\Lambda_{v_2}$', fontsize=12)
ax_lam.legend(fontsize=10)
fig_lam.tight_layout()
fig_lam.savefig('apai_lambda_v2.png', dpi=150)
plt.close(fig_lam)

# Row 2, (1,2): EKF uncertainty P_v3
fig_p, ax_p = plt.subplots(figsize=(6, 4))
for res in results:
    color = res['color']
    ax_p.plot(t, res['P_hist'][:, 2], color=color, linewidth=2, label=res['name'])
obs_times_v3 = [step + 1 for step in range(T) if step % 3 == 2]
for obs_t in obs_times_v3:
    ax_p.axvline(obs_t, color='gray', alpha=0.2, linewidth=0.8)
ax_p.set_title('EKF Uncertainty $P_{v_3}(t)$', fontsize=13)
ax_p.set_xlabel('Interaction step', fontsize=12)
ax_p.set_ylabel('$P_{v_3}(t)$', fontsize=12)
ax_p.legend(fontsize=10)
fig_p.tight_layout()
fig_p.savefig('apai_uncertainty_v3.png', dpi=150)
plt.close(fig_p)

print("\nAtrophy thresholds c* per domain:")
for v in range(3):
    print(f"  {domains[v]}: c* = {c_star_vec[v]:.3f}")
print("\nFinal Capability and EKF Estimate Summary:")
print(f"{'Scenario':<12} {'Domain':<14} {'True c':<10} {'EKF c_hat':<10}")
for res in results:
    for v in range(3):
        c_true_final = res['c_true_hist'][-1, v]
        c_hat_final = res['c_hat_hist'][-1, v]
        print(f"{res['name']:<12} {domains[v]:<14} {c_true_final:<10.3f} {c_hat_final:<10.3f}")

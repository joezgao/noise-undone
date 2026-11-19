"""Recompute the README's model numbers from the weights shipped in noise-undone.html.

The training scripts were lost (see README), so training cannot be rerun. This checks
what can still be checked: parameter count, held-out MSE on 200k fresh draws, and the
distance from 4,000 DDIM / DDPM samples to the knot's centre line, plus coverage of
every segment of the knot. Sampling noise moves the numbers by about +-0.001.

    python3 check/verify_readme.py [noise-undone.html] [seed]     # needs NumPy, ~1 min
"""
import json, base64, sys, numpy as np
html = open(sys.argv[1] if len(sys.argv) > 1 else 'noise-undone.html').read()
line = [l for l in html.splitlines() if l.startswith('const MODEL = {')][0]
m = json.loads(line[line.index('{'):line.rindex('}')+1])
raw = np.frombuffer(base64.b64decode(m['b64']), dtype='<f2')
P = {t['n']: raw[t['o']//2: t['o']//2 + int(np.prod(t['s']))].astype(np.float64).reshape(t['s']) for t in m['tensors']}
nparams = sum(v.size for v in P.values())
F = np.array(m['freqs']); TF = np.array(m['tfreqs']); abar = np.array(m['abar']); betas = np.array(m['betas'])
def eps(x, t):  # x (n,3), t (n,) ints
    f = np.concatenate([x, np.sin(x[:, :, None]*F).reshape(len(x), -1), np.cos(x[:, :, None]*F).reshape(len(x), -1)], 1)
    a = t[:, None]*TF
    inp = np.concatenate([f, np.sin(a), np.cos(a)], 1)
    silu = lambda u: u/(1+np.exp(-u))
    h = silu(inp @ P['W1'].T + P['b1'])
    h = h + silu(h @ P['W2'].T + P['b2'])
    h = h + silu(h @ P['W3'].T + P['b3'])
    return h @ P['W4'].T + P['b4']
def curve(s):
    return np.stack([np.sin(s)+2*np.sin(2*s), np.cos(s)-2*np.cos(2*s), -np.sin(3*s)], -1)*np.array([.5, .5, .8])
def data(n, rng):
    s = rng.uniform(0, 2*np.pi, n); c = curve(s)
    d = np.stack([np.cos(s)+4*np.cos(2*s), -np.sin(s)+4*np.sin(2*s), -3*np.cos(3*s)], -1)*np.array([.5, .5, .8])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a = np.cross(d, [0.3, 0.5, 0.8]); a /= np.linalg.norm(a, axis=1, keepdims=True); b = np.cross(d, a)
    r = 0.06*np.sqrt(rng.uniform(0, 1, n)); th = rng.uniform(0, 2*np.pi, n)
    return c + r[:, None]*(np.cos(th)[:, None]*a + np.sin(th)[:, None]*b)
line_pts = curve(np.linspace(0, 2*np.pi, 40000, endpoint=False))
def dist(x):
    out = []
    for i in range(0, len(x), 500):
        out.append(np.sqrt(((x[i:i+500, None, :]-line_pts[None])**2).sum(-1)).min(1))
    return np.concatenate(out)
def sample(n, kind, rng):
    x = rng.standard_normal((n, 3))
    for t in range(100, 0, -1):
        e = eps(x, np.full(n, t)); ab, abp = abar[t], abar[t-1]
        if kind == 'ddim':
            x0 = np.clip((x-np.sqrt(1-ab)*e)/np.sqrt(ab), -3, 3); x = np.sqrt(abp)*x0+np.sqrt(1-abp)*e
        else:
            b = betas[t-1]; x = (x-b/np.sqrt(1-ab)*e)/np.sqrt(1-b)
            if t > 1: x += np.sqrt(b*(1-abp)/(1-ab))*rng.standard_normal((n, 3))
    return x
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
print('params', nparams, 'json', m['params'])
tot = 0.0; N = 200000
for i in range(0, N, 20000):
    x0 = data(20000, rng); t = rng.integers(1, 101, 20000); e = rng.standard_normal((20000, 3))
    xt = np.sqrt(abar[t])[:, None]*x0+np.sqrt(1-abar[t])[:, None]*e
    tot += ((eps(xt, t)-e)**2).sum()
print('heldout_mse %.4f' % (tot/(N*3)))
print('real_median %.4f' % np.median(dist(data(4000, rng))))
for k in ('ddim', 'ddpm'):
    d = dist(x := sample(4000, k, rng))
    seg = np.bincount(np.concatenate([np.sqrt(((x[i:i+500, None]-line_pts[None])**2).sum(-1)).argmin(1) for i in range(0, 4000, 500)])*72//len(line_pts), minlength=72)
    print('%s median %.4f within0.12 %.2f%% max %.3f  emptiest of 72 segments: %d samples' % (k, np.median(d), 100*(d < 0.12).mean(), d.max(), seg.min()))

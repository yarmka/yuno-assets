#!/usr/bin/env python3
"""Yuno audio pipeline: raw recordings -> clean, trimmed, loudness-matched, compressed (Ogg Vorbis) game assets.

Everything is done in float32 at the NATIVE sample rate of each recording (no resampling of the finished sound, so no
quality is lost there); the only lossy step is the final Vorbis encode (quality 6-7 = ~190-230 kbps stereo, transparent).
"""
import os, re, sys, subprocess, json, shutil
import numpy as np
import scipy.io.wavfile as wf
import scipy.signal as sg

RAW = '/home/claude/work/snd_raw/MGC_CRASH_LOG'
OUT = '/home/claude/work/Yuno/audio'
TMP = '/home/claude/work/tmp_wav'
os.makedirs(TMP, exist_ok=True)

# ------------------------------------------------------------------ io helpers
def find(key):
    hits = [f for f in os.listdir(RAW) if key in f]
    assert len(hits) == 1, (key, hits)
    return os.path.join(RAW, hits[0])

def load(key):
    sr, x = wf.read(find(key))
    x = x.astype(np.float32) / 32768.0
    if x.ndim == 1:
        x = x[:, None]
    return sr, x            # (n, ch)

def mono(x):
    return x.mean(axis=1, keepdims=True) if x.shape[1] > 1 else x

def db(v):
    return 20 * np.log10(max(float(v), 1e-9))

def peak(x):
    return float(np.abs(x).max())

def norm_peak(x, target_db=-1.0):
    p = peak(x)
    return x * (10 ** (target_db / 20) / p) if p > 0 else x

def norm_rms(x, target_db):
    r = float(np.sqrt((x ** 2).mean()))
    return x * (10 ** (target_db / 20) / r)

def env_rms(x, sr, win=0.01):
    m = np.abs(mono(x)[:, 0])
    w = max(1, int(win * sr))
    return np.sqrt(np.convolve(m ** 2, np.ones(w) / w, 'same'))

def trim(x, sr, thr_db=-55, head_pad=0.004, tail_pad=0.03):
    """cut leading / trailing silence (relative to the loudest point)"""
    e = env_rms(x, sr)
    thr = e.max() * 10 ** (thr_db / 20)
    idx = np.where(e > thr)[0]
    a = max(0, idx[0] - int(head_pad * sr))
    b = min(len(x), idx[-1] + int(tail_pad * sr))
    return x[a:b]

def fade(x, sr, fin=0.002, fout=0.02):
    x = x.copy()
    n_in = int(fin * sr); n_out = int(fout * sr)
    if n_in > 0:
        x[:n_in] *= np.linspace(0, 1, n_in)[:, None]
    if n_out > 0:
        x[-n_out:] *= np.cos(np.linspace(0, np.pi / 2, n_out))[:, None] ** 2
    return x

def resample(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    from math import gcd
    g = gcd(sr_from, sr_to)
    return sg.resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)

def pitch_speed(x, factor):
    """play faster (factor > 1: higher pitch, shorter) using a high quality polyphase resampler"""
    from fractions import Fraction
    f = Fraction(factor).limit_denominator(200)
    return sg.resample_poly(x, f.denominator, f.numerator, axis=0).astype(np.float32)

def lowpass(x, sr, fc, order=4):
    b, a = sg.butter(order, fc / (sr / 2), 'low')
    return sg.filtfilt(b, a, x, axis=0).astype(np.float32)

def highpass(x, sr, fc, order=2):
    b, a = sg.butter(order, fc / (sr / 2), 'high')
    return sg.filtfilt(b, a, x, axis=0).astype(np.float32)

def softclip(x, drive=1.0):
    return np.tanh(x * drive) / np.tanh(drive)

def loop_xfade(x, sr, xf=0.08):
    """make a seamless loop: the last `xf` seconds are cross-faded into the first ones, then removed"""
    n = int(xf * sr)
    head = x[:n]; tail = x[-n:]
    w = np.linspace(0, 1, n)[:, None]
    # equal-power
    mix = head * np.sin(w * np.pi / 2) + tail * np.cos(w * np.pi / 2)
    return np.concatenate([mix, x[n:-n]], axis=0)

def write_wav(path, x, sr, bits=16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    wf.write(path, sr, (x * 32767.0).astype(np.int16) if x.shape[1] > 1 else (x[:, 0] * 32767.0).astype(np.int16))

def encode_ogg(x, sr, out_path, q=6.0):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = os.path.join(TMP, os.path.basename(out_path) + '.wav')
    wf.write(tmp, sr, np.clip(x, -1, 1).astype(np.float32) if x.shape[1] > 1 else np.clip(x[:, 0], -1, 1).astype(np.float32))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'libvorbis', '-q:a', str(q), out_path], check=True)
    os.remove(tmp)

REPORT = []
def emit(rel, x, sr, q=6.0, keep_stereo=False, src=None, note=''):
    """finish one sound: mono-ise (unless asked), write ogg, remember the numbers"""
    if not keep_stereo:
        x = mono(x)
    path = os.path.join(OUT, rel)
    encode_ogg(x, sr, path, q)
    REPORT.append({'file': rel, 'sr': sr, 'ch': x.shape[1], 'dur': round(len(x) / sr, 3), 'bytes': os.path.getsize(path),
                   'peak_db': round(db(peak(x)), 1), 'src': src, 'note': note})
    return path

# ------------------------------------------------------------------ weapons
def distance_variant(x, sr, level):
    """same recipe as the old procedural WeaponSfx._distance_fx: air absorbs the highs, mountains / buildings echo"""
    x = mono(x)[:, 0].astype(np.float64)
    fc = 3600.0 if level == 1 else 900.0
    passes = 1 if level == 1 else 3
    a = np.exp(-2 * np.pi * fc / sr)
    for _ in range(passes):
        x = sg.lfilter([1 - a], [1, -a], x)
    extra = int((0.55 if level == 1 else 1.25) * sr)
    out = np.zeros(len(x) + extra)
    out[:len(x)] = x
    taps = [(0.16, 0.32), (0.37, 0.18)] if level == 1 else [(0.23, 0.55), (0.51, 0.40), (0.84, 0.27), (1.20, 0.15)]
    for d, g in taps:
        k = int(d * sr)
        out[k:k + len(x)] += x * g
    out = out / max(np.abs(out).max(), 1e-9)
    return fade(out[:, None].astype(np.float32), sr, 0.001, 0.15)

def build_weapons():
    # M4
    sr, x = load('اطلاق m4')
    x = trim(x, sr, -60, 0.002, 0.02)
    m4 = fade(norm_peak(x, -0.8), sr, 0.0005, 0.03)
    emit('sfx/weapons/m4_shot.ogg', m4, sr, 7, keep_stereo=True, src='اطلاق m4')
    for lvl, nm in ((1, 'mid'), (2, 'far')):
        v = norm_peak(distance_variant(m4, sr, lvl), -1.0)
        emit(f'sfx/weapons/m4_shot_{nm}.ogg', v, sr, 6, src='baked distance')
    # Sniper (3.25 s file, 1.8 s of it silence)
    sr, x = load('طلقه سنايبر')
    x = trim(x, sr, -62, 0.002, 0.15)
    sn = fade(norm_peak(x, -0.8), sr, 0.0005, 0.25)
    emit('sfx/weapons/sniper_shot.ogg', sn, sr, 7, keep_stereo=True, src='طلقه سنايبر')
    for lvl, nm in ((1, 'mid'), (2, 'far')):
        v = norm_peak(distance_variant(sn, sr, lvl), -1.0)
        emit(f'sfx/weapons/sniper_shot_{nm}.ogg', v, sr, 6, src='baked distance')
    # Shotgun (no shotgun weapon in the game yet: ready for it)
    sr, x = load('اطلاق شدقن')
    x = trim(x, sr, -62, 0.002, 0.2)
    emit('sfx/weapons/shotgun_shot.ogg', fade(norm_peak(x, -0.8), sr, 0.0005, 0.3), sr, 7, keep_stereo=True, src='اطلاق شدقن')
    # bullet whiz-by (two takes)
    for i, key in enumerate(['صوت مرور الرصاصه', 'مرور رصاصه 2'], 1):
        sr, x = load(key)
        x = trim(x, sr, -55, 0.002, 0.05)
        # keep their natural relative level (take 2 is quieter): the close / far split in code uses both
        x = fade(x, sr, 0.0005, 0.06)
        emit(f'sfx/weapons/whiz_{i}.ogg', x, sr, 6, src=key)
    # impacts
    sr, x = load('رصاصه في الحديد')
    x = trim(x, sr, -55, 0.002, 0.05)
    metal = fade(norm_peak(x, -1.5), sr, 0.0005, 0.05)
    emit('sfx/weapons/impact_metal.ogg', metal, sr, 6, src='رصاصه في الحديد')
    # generic hard surface (concrete / brick): the same hit, ring removed (low-pass), a little lower & shorter
    h = lowpass(mono(metal), sr, 2600, 4)
    h = pitch_speed(h, 0.86)
    h = fade(norm_peak(h[: int(0.32 * sr)], -2.5), sr, 0.0005, 0.08)
    emit('sfx/weapons/impact_hard.ogg', h, sr, 6, src='derived from رصاصه في الحديد')
    sr, x = load('ضربه رصاصه على سياره')
    x = trim(x, sr, -55, 0.004, 0.05)
    emit('sfx/weapons/impact_car.ogg', fade(norm_peak(x, -1.5), sr, 0.0005, 0.08), sr, 6, src='ضربه رصاصه على سياره')

# ------------------------------------------------------------------ people
def build_people():
    def one(rel, key, target=-1.0, thr=-55, tail=0.03, head=0.01, q=6, rate=None):
        sr, x = load(key)
        x = mono(x)
        x = trim(x, sr, thr, head, tail)
        x = fade(norm_peak(x, target), sr, 0.003, min(0.05, tail))
        emit(rel, x, sr, q, src=key)
    # player
    one('sfx/player/hurt_1.ogg', 'ضرر الاعب')
    one('sfx/player/hurt_2.ogg', 'دمج اللاعب2')
    one('sfx/player/death.ogg', 'موت اللاعب')
    one('sfx/player/scream_fall_1.ogg', 'يسقط.wav', tail=0.05)
    one('sfx/player/scream_fall_2.ogg', 'يسقط2.wav', tail=0.05)
    one('sfx/player/scream_2.ogg', 'صراخ 2', tail=0.05)
    # NPC men / women
    one('sfx/npc/scream_m_1.ogg', 'صراخ الامبيسي.wav', tail=0.05)
    one('sfx/npc/scream_f_1.ogg', 'صراخ الامبيسي البنت', tail=0.05)
    one('sfx/npc/scream_f_2.ogg', 'صراخ الامبيسي بنت 2', tail=0.05)
    one('sfx/npc/scream_f_3.ogg', 'صراخ بنت امبيسي2', tail=0.05)
    one('sfx/npc/scream_f_4.ogg', 'صراخ بنت امبيسي3', tail=0.05)
    one('sfx/npc/death_f.ogg', 'موت الامبيسي بنت')

# ------------------------------------------------------------------ vehicles
def period_loop(x, sr, f_lo=80, f_hi=1500):
    """cut a periodic tone to a whole number of periods (zero-crossing aligned) so it loops without a click"""
    x1 = x[:, 0]
    ac = np.correlate(x1 - x1.mean(), x1 - x1.mean(), 'full')[len(x1) - 1:]
    a, b = int(sr / f_hi), int(sr / f_lo)
    k = a + int(np.argmax(ac[a:b]))
    n_per = int(round(len(x1) / k))
    n = k * max(n_per, 1)
    return x[:n], sr / k

def build_vehicles():
    # --- door: open = the recording, close = the 2nd knock (the door body) lower & duller
    sr, x = load('فتح باب سياره')
    x = mono(x)
    x = trim(x, sr, -50, 0.003, 0.04)
    door_open = fade(norm_peak(x, -1.5), sr, 0.001, 0.05)
    emit('sfx/car/door_open.ogg', door_open, sr, 6, src='فتح باب سياره')
    seg = x[int(0.17 * sr):]                                    # from the 2nd event on (door swinging / latch)
    seg = trim(seg, sr, -50, 0.004, 0.05)
    seg = pitch_speed(seg, 0.88)
    seg = lowpass(seg, sr, 5200, 2)
    # a short low "thump" of the door body (synthesised sine sweep, very quiet) gives a closing door its weight
    n = int(0.12 * sr); t = np.arange(n) / sr
    thump = (np.sin(2 * np.pi * (95 - 40 * t / 0.12) * t) * np.exp(-t * 28) * 0.5).astype(np.float32)[:, None]
    seg = seg.copy(); seg[:n] += thump[:len(seg[:n])]
    emit('sfx/car/door_close.ogg', fade(norm_peak(seg, -2.0), sr, 0.001, 0.06), sr, 6, src='derived from فتح باب سياره')

    # --- heavy crash (new top tier)
    sr, x = load('اصتدام سياره')
    x = trim(mono(x), sr, -58, 0.003, 0.2)
    emit('sfx/car/crash_xl.ogg', fade(norm_peak(x, -1.0), sr, 0.0005, 0.3), sr, 6, src='اصتدام سياره')

    # --- scrape loop
    sr, x = load('احتكاك سياره ب اشياء')
    x = loop_xfade(mono(x), sr, 0.12)
    emit('sfx/car/scrape_loop.ogg', norm_peak(x, -2.0), sr, 6, src='احتكاك سياره ب اشياء', note='seamless loop')

    # --- tyre chirps
    sr, x = load('تفحيط كفرات')
    x = trim(mono(x), sr, -50, 0.004, 0.04)
    emit('sfx/car/tyre_burn.ogg', fade(norm_peak(x, -1.5), sr, 0.002, 0.08), sr, 6, src='تفحيط كفرات')
    sr, x = load('تفحيطه لفه')
    x = trim(mono(x), sr, -50, 0.004, 0.04)
    emit('sfx/car/tyre_turn.ogg', fade(norm_peak(x, -1.5), sr, 0.002, 0.08), sr, 6, src='تفحيطه لفه')

    # --- horns: loops that must stay WAV (Car.gd loops an AudioStreamWAV); whole number of periods = no click
    sr, x = load('صوت بوري سياره.wav')
    x = mono(x)
    xl, f0 = period_loop(x, sr)
    old = wf.read('/home/claude/work/Yuno/audio/car/horn.wav')
    ox = old[1].astype(np.float32) / 32768.0
    old_rms = float(np.sqrt((ox ** 2).mean()))
    xl = xl * (old_rms / float(np.sqrt((xl ** 2).mean())))        # same loudness as the horn it replaces
    xl = np.clip(xl, -0.98, 0.98)
    write_wav(os.path.join(OUT, 'car/horn.wav'), xl, sr)
    REPORT.append({'file': 'car/horn.wav', 'sr': sr, 'ch': 1, 'dur': round(len(xl) / sr, 3), 'bytes': os.path.getsize(os.path.join(OUT, 'car/horn.wav')), 'peak_db': round(db(peak(xl)), 1), 'src': 'بوري سياره', 'note': f'loop of {len(xl)/ (sr/f0):.0f} periods, f0={f0:.0f} Hz (kept WAV: Car.gd loops it as AudioStreamWAV)'})
    sr, x = load('بوري سياره npc')
    x = mono(x)
    xl, f0 = period_loop(x, sr, 60, 1500)
    xl = np.clip(norm_peak(xl, -3.0), -0.98, 0.98)
    write_wav(os.path.join(OUT, 'sfx/car/horn_npc.wav'), xl, sr)
    REPORT.append({'file': 'sfx/car/horn_npc.wav', 'sr': sr, 'ch': 1, 'dur': round(len(xl) / sr, 3), 'bytes': os.path.getsize(os.path.join(OUT, 'sfx/car/horn_npc.wav')), 'peak_db': -3.0, 'src': 'بوري سياره npc', 'note': f'loop, f0={f0:.0f} Hz'})

    # --- engine idle (seamless loop), same level as the idle it replaces
    sr, x = load('engine idle sound')
    x = loop_xfade(mono(x), sr, 0.15)
    oi = wf.read('/home/claude/work/Yuno/audio/car/eng_idle.wav')[1].astype(np.float32) / 32768.0
    x = norm_rms(x, db(np.sqrt((oi ** 2).mean())))
    x = np.clip(x, -0.98, 0.98)
    emit('car/eng_idle.ogg', x, sr, 6, src='engine idle sound', note='seamless loop, loudness matched to the old idle')

    # --- "da3sa" = flooring the throttle: a steady ~120 Hz rev, heavily clipped in the source -> tamed
    sr, x = load('دعسه سياره')
    x = mono(x)[int(0.4 * sr):int(6.4 * sr)]
    x = lowpass(x, sr, 7000, 2)           # take the harsh clipping fizz off the top
    x = softclip(x, 0.9)
    x = loop_xfade(x, sr, 0.25)
    emit('car/eng_rev_loop.ogg', norm_peak(x, -6.0), sr, 6, src='دعسه سياره', note='source is clipped (30% of samples); softened')

# ------------------------------------------------------------------ helicopter
def build_heli():
    sr, x = load('مروحيه')
    x = trim(mono(x), sr, -66, 0.01, 0.4)
    emit('sfx/misc/heli_startstop.ogg', fade(norm_peak(x, -1.5), sr, 0.002, 0.4), sr, 5, src='ايقاف وتشغيل مروحيه')

# ------------------------------------------------------------------ footsteps
def find_steps(x, sr, thr_rel, min_gap=0.25):
    m = mono(x)[:, 0]
    y = highpass(m[:, None], sr, 250)[:, 0]
    env = np.abs(sg.hilbert(y))
    w = int(0.004 * sr)
    env = np.convolve(env, np.ones(w) / w, 'same')
    pk, _ = sg.find_peaks(env, height=env.max() * thr_rel, distance=int(min_gap * sr))
    return pk, env

def cut_step(x, sr, p, pre=0.022, length=0.42, next_p=None):
    a = max(0, p - int(pre * sr))
    b = min(len(x), p + int(length * sr))
    if next_p is not None:
        b = min(b, next_p - int(0.03 * sr))
    return x[a:b]

def finish_step(s, sr, target=-1.0, fout=0.12):
    s = fade(s, sr, 0.0008, fout)
    return norm_peak(s, target)

def layer(a, b, sr, offset):
    n = max(len(a), len(b) + int(offset * sr))
    o = np.zeros((n, 1), np.float32)
    o[:len(a)] += a
    k = int(offset * sr)
    o[k:k + len(b)] += b
    return o

def low_thump(sr, f0=105, dur=0.16, gain=0.55):
    n = int(dur * sr); t = np.arange(n) / sr
    return (np.sin(2 * np.pi * (f0 - 35 * t / dur) * t) * np.exp(-t * 22) * gain).astype(np.float32)[:, None]

def build_steps(key, group, thr_rel, hard, order_keep=None):
    sr, x = load(key)
    x = mono(x)
    pk, env = find_steps(x, sr, thr_rel)
    # discard the first (half-recorded) step and any that is too close to the file end
    pk = [p for p in pk if p > int(0.05 * sr) and p < len(x) - int(0.45 * sr)]
    pk_all = list(pk)
    slices = []
    for i, p in enumerate(pk_all):
        nxt = pk_all[i + 1] if i + 1 < len(pk_all) else None
        s = cut_step(x, sr, p, 0.022, 0.42, nxt)
        e120 = float(np.sqrt((s[:int(0.12 * sr)] ** 2).mean()))
        slices.append((e120, p, s))
    slices.sort(key=lambda t: t[0])                 # quiet ... loud
    n = len(slices)
    lo = slices[: max(3, n // 4)]
    mid = slices[n // 4: 3 * n // 4]
    hi = slices[-max(3, n // 4):]
    pick = lambda lst, k: [lst[int(round(i * (len(lst) - 1) / max(k - 1, 1)))] for i in range(k)]
    walk = pick(mid, 3)
    run = pick(hi, 3)
    sneak = pick(lo, 3)
    info = {'steps_found': n}
    ext = '.ogg'
    def out(gait, i, s):
        encode_ogg(s, sr, os.path.join(OUT, f'steps/{group}_{gait}_{i}{ext}'), 6)
        path = os.path.join(OUT, f'steps/{group}_{gait}_{i}{ext}')
        REPORT.append({'file': f'steps/{group}_{gait}_{i}{ext}', 'sr': sr, 'ch': 1, 'dur': round(len(s) / sr, 3), 'bytes': os.path.getsize(path), 'peak_db': round(db(peak(s)), 1), 'src': key, 'note': gait})
    for i, (_, p, s) in enumerate(walk, 1):
        out('walk', i, finish_step(s, sr, -1.0, 0.14))
    for i, (_, p, s) in enumerate(run, 1):
        r = pitch_speed(s, 1.10)                       # a run is quicker and harder
        r = r[: int(0.34 * sr)]
        out('run', i, finish_step(r, sr, -1.0, 0.10))
    for i, (_, p, s) in enumerate(sneak, 1):
        q = lowpass(s, sr, 3800, 2)                    # careful steps: the sole barely touches, so the top end is soft
        q = q[: int(0.42 * sr)]
        out('sneak', i, finish_step(q, sr, -1.0, 0.16))
    # landing: both feet together (two hits 12 ms apart) + a short low body thump
    land_src = pick(hi, 3)
    for i, (_, p, s) in enumerate(land_src, 1):
        other = hi[(i) % len(hi)][2]
        l = layer(s, other * 0.8, sr, 0.012)
        l = l[: int(0.46 * sr)]
        l = l.copy(); th = low_thump(sr, 105 if hard else 85, 0.16, 0.6 if hard else 0.8)
        l[:len(th)] += th * peak(l)
        out('land', i, finish_step(l, sr, -1.0, 0.18))
    # jump push-off: the toe only, short and a little brighter
    jump_src = pick(mid, 3)
    for i, (_, p, s) in enumerate(jump_src, 1):
        j = pitch_speed(s, 1.06)[: int(0.22 * sr)]
        out('jump', i, finish_step(j, sr, -1.0, 0.09))
    return info

def build_all_steps():
    info = {}
    info['asphalt'] = build_steps('مشي على ازفلت', 'asphalt', 0.10, True)
    info['grass'] = build_steps('مشي على العشب', 'grass', 0.07, False)
    return info


if __name__ == '__main__':
    steps_info = None
    what = sys.argv[1:] or ['weapons', 'people', 'vehicles', 'heli', 'steps']
    if 'weapons' in what: build_weapons()
    if 'people' in what: build_people()
    if 'vehicles' in what: build_vehicles()
    if 'heli' in what: build_heli()
    if 'steps' in what: steps_info = build_all_steps()
    with open('/home/claude/work/audio_report.json', 'w') as f:
        json.dump({'files': REPORT, 'steps': steps_info}, f, ensure_ascii=False, indent=1)
    tot = sum(r['bytes'] for r in REPORT)
    print(len(REPORT), 'files', tot // 1024, 'KB')
